from flash_attn_triton import attention
import numpy as np
import torch
from flash_attn.flash_attn_interface import \
    flash_attn_func


DEVICE="cuda:0"

def test_op(Z=32, H=32, N_CTX=8192, HEAD_DIM=128, causal=False, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 1 / np.sqrt(HEAD_DIM)
    # reference implementation

    tri_out = attention(q, k, v, causal, sm_scale).half()
    # dout = torch.ones_like(q)
    dout = torch.randn_like(q)

    # print(f"before backward: {q.grad=}")
    tri_out.backward(dout)
    # print(f"after backward: {q.grad=}")
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    ref_out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=sm_scale,
        causal=causal
    ).transpose(1, 2)

    ref_out.backward(dout)
    # print(f"after backward: {q.grad=}")
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    rtol=0
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)
    print("tri backward equals ref. OK!")

    sp_dv, sp_dk = torch.zeros_like(v), torch.zeros_like(k)
    num_samples=1
    for i in range(num_samples):
        subset_size = 2048
        mask = torch.randperm(N_CTX, device=q.device)[:subset_size].sort().values

        # mask = (torch.rand(N_CTX) > 0.9)
        # subset_size = mask.sum()

        masked_q = q[:, :, mask].detach().clone().requires_grad_()
        sparse_out = attention(masked_q, k, v, causal, sm_scale).half()

        # print(f"before backward: {masked_q.grad=}")
        sparse_out.backward(dout[:, :, mask])
        # sparse_out.backward(dout)
        # print(f"after backward: {masked_q.grad=}")
        sparse_dv, v.grad = v.grad.clone(), None
        sparse_dk, k.grad = k.grad.clone(), None
        sparse_dq, masked_q.grad = masked_q.grad.clone(), None

        # print(sp_dv)
        print(i)
        if torch.any(torch.isinf(sparse_dv)):
            exit("found infs")

        sp_dv += sparse_dv / (subset_size/N_CTX) / num_samples
        sp_dk += sparse_dk / (subset_size/N_CTX) / num_samples

    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol=0
    print(f"{tri_dv.size()=} {sp_dv.size()=}")
    print(f"{tri_dk.size()=} {sp_dk.size()=}")
    print(f"{tri_dq[:, :, mask].size()=} {sparse_dq.size()=}")

    # print(f"{tri_dq[0, 0, 0:2]=} {sparse_dq[0, 0, 0:2]=}")
    # print(f"{tri_dq[0, 0, 0:2] / sparse_dq[0, 0, 0:2]=}")

    def test_close(a, b, name):
        print(f"{a.size()=} {b.size()=} {name=}")
        diff = (a - b).abs()
        cos = torch.sum(a * b, dim=-1, keepdim=True)
        cos = cos / (
            (a ** 2).sum(dim=-1, keepdim=True).sqrt() * \
            (b ** 2).sum(dim=-1, keepdim=True).sqrt() + 1e-8
        )
        cos = cos.mean()
        print(f"{name} {diff.mean()=} {diff.amax()=} {diff.std()=} {cos=}")

    test_close(tri_dv, sp_dv, "dv")
    test_close(tri_dk, sp_dk, "dk")
    test_close(tri_dq[:, :, mask], sparse_dq, "dq")

    # torch.testing.assert_close(tri_dv, sp_dv, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(tri_dk, sp_dk, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(tri_dq[:, :, mask], sparse_dq, atol=1e-2, rtol=rtol)


def test_causal(Z=4, H=32, N_CTX=1024, HEAD_DIM=128, causal=True, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 1 / np.sqrt(HEAD_DIM)
    # reference implementation

    # mask = torch.arange(N_CTX, device=q.device).view(1, N_CTX).repeat(Z, 1)
    mask = torch.randperm(N_CTX, device=q.device)[:256].sort().values
    tri_out = attention(q[:, :, mask], k, v, mask, causal, sm_scale).half()
    dout = torch.randn_like(q)

    # tri_out.backward(dout)
    # print(f"after backward: {q.grad=}")
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dq, q.grad = q.grad.clone(), None

    ref_out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        softmax_scale=sm_scale,
        causal=causal
    ).transpose(1, 2)
    ref_out.backward(dout)

    ref_out = ref_out[:, :, mask]

    # print(f"after backward: {q.grad=}")
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    rtol=0
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(tri_dv, ref_dv, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(tri_dk, ref_dk, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(tri_dq, ref_dq, atol=1e-2, rtol=rtol)
    print("tri backward equals ref. OK!")
    exit("exiting early")

    sp_dv, sp_dk = torch.zeros_like(v), torch.zeros_like(k)
    num_samples=1
    for i in range(num_samples):
        subset_size = 2048
        mask = torch.randperm(N_CTX, device=q.device)[:subset_size].sort().values

        # mask = (torch.rand(N_CTX) > 0.9)
        # subset_size = mask.sum()

        masked_q = q[:, :, mask].detach().clone().requires_grad_()
        sparse_out = attention(masked_q, k, v, causal, sm_scale).half()

        # print(f"before backward: {masked_q.grad=}")
        sparse_out.backward(dout[:, :, mask])
        # sparse_out.backward(dout)
        # print(f"after backward: {masked_q.grad=}")
        sparse_dv, v.grad = v.grad.clone(), None
        sparse_dk, k.grad = k.grad.clone(), None
        sparse_dq, masked_q.grad = masked_q.grad.clone(), None

        # print(sp_dv)
        print(i)
        if torch.any(torch.isinf(sparse_dv)):
            exit("found infs")

        sp_dv += sparse_dv / (subset_size/N_CTX) / num_samples
        sp_dk += sparse_dk / (subset_size/N_CTX) / num_samples

    # Relative tolerance workaround for known hardware limitation of CDNA2 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    rtol=0
    print(f"{tri_dv.size()=} {sp_dv.size()=}")
    print(f"{tri_dk.size()=} {sp_dk.size()=}")
    print(f"{tri_dq[:, :, mask].size()=} {sparse_dq.size()=}")

    # print(f"{tri_dq[0, 0, 0:2]=} {sparse_dq[0, 0, 0:2]=}")
    # print(f"{tri_dq[0, 0, 0:2] / sparse_dq[0, 0, 0:2]=}")

    def test_close(a, b, name):
        print(f"{a.size()=} {b.size()=} {name=}")
        diff = (a - b).abs()
        cos = torch.sum(a * b, dim=-1, keepdim=True)
        cos = cos / (
            (a ** 2).sum(dim=-1, keepdim=True).sqrt() * \
            (b ** 2).sum(dim=-1, keepdim=True).sqrt() + 1e-8
        )
        cos = cos.mean()
        print(f"{name} {diff.mean()=} {diff.amax()=} {diff.std()=} {cos=}")

    test_close(tri_dv, sp_dv, "dv")
    test_close(tri_dk, sp_dk, "dk")
    test_close(tri_dq[:, :, mask], sparse_dq, "dq")

    # torch.testing.assert_close(tri_dv, sp_dv, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(tri_dk, sp_dk, atol=1e-2, rtol=rtol)
    # torch.testing.assert_close(tri_dq[:, :, mask], sparse_dq, atol=1e-2, rtol=rtol)
if __name__ == "__main__":
    test_causal()
    test_op()
