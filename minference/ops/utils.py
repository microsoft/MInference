import os
import torch

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def use_triton():
    return torch.version.hip is not None or os.getenv("FORCE_TRITON", "0") == "1"


def check_correctness_by_row(
    seq_len,
    tensor_var,
    ref_tensor_var,
    tensor_name,
    ATOL=1e-2,
    RTOL=1e-2,
):
    if not torch.allclose(tensor_var, ref_tensor_var, atol=ATOL, rtol=RTOL):
        for h in range(tensor_var.shape[2]):
            for i in range(seq_len):
                tensor_var_row = tensor_var[:, i, h]
                ref_tensor_var_row = ref_tensor_var[:, i, h]

                if not torch.allclose(tensor_var_row, ref_tensor_var_row, atol=ATOL, rtol=RTOL):
                    print('-' * 60 + '\n')
                    print(f"Mismatched {tensor_name} at Head {h}, Row {i}:\n")
                    print(f"Computed:\n{tensor_var_row}\n")
                    print(f"Ref:\n{ref_tensor_var_row}\n")
                    
                    max_diff = torch.max(torch.abs(tensor_var_row - ref_tensor_var_row))
                    print(f"Maximal difference: {max_diff.item()}\n")
        return False
    else:
        print(f"All {tensor_name} values are correct within the specified tolerance.")
        return True

def check_correct_rate(
    tensor_var,
    ref_tensor_var,
    ATOL=1e-2,
    RTOL=1e-2,
):
    assert len(tensor_var.shape) == 4, "Input tensor must be 3D (B, N, H, D)"
    assert tensor_var.shape == ref_tensor_var.shape, (
        "Input and reference tensors must have the same shape"
    )
    bsz, seq_len, num_heads, _ = tensor_var.shape

    # Boolean mask of element-wise closeness
    elem_close = torch.isclose(tensor_var, ref_tensor_var, atol=ATOL, rtol=RTOL)

    # A row “matches” only if *all* its D elements are close
    row_matches = elem_close.all(dim=-1)        # shape (B, N, H)

    # Count rows that do *not* match
    num_mismatching = (~row_matches).sum().item()
    num_mismatching_prop = num_mismatching / (bsz * seq_len * num_heads)
    return 1 - num_mismatching_prop

def check_by_correct_rate(
    tensor_var,
    ref_tensor_var,
    ATOL=1e-2,
    RTOL=1e-2,
    threshold=0.99
):
    """
    Check if the tensor_var is correct by comparing it with ref_tensor_var.
    Returns True if the correctness rate is above 0.99, otherwise False.
    """
    correctness_rate = check_correct_rate(tensor_var, ref_tensor_var, ATOL, RTOL)
    return correctness_rate >= threshold