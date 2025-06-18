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
    else:
        print(f"All {tensor_name} values are correct within the specified tolerance.")