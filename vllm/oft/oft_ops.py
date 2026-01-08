import torch

def _pytorch_skew_symmetric(vec: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Constructs a skew-symmetric matrix from a vector of upper triangular elements.
    vec: (batch_size, n_elements) or (n_elements,)
    block_size: Dimension of the square matrix
    """
    # Handle single vector case by unsqueezing
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    
    batch_size = vec.shape[0]
    
    # Pre-compute indices if not cached (for functional statelessness we compute on fly or cache globally)
    # For efficiency in vLLM, we compute them on the fly using the device of vec
    rows, cols = torch.triu_indices(block_size, block_size, 1, device=vec.device)
    
    matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)
    matrix[:, rows, cols] = vec
    matrix = matrix - matrix.transpose(-2, -1)
    
    return matrix

def _cayley_batch(
    Q: torch.Tensor, 
    block_size: int, 
    use_cayley_neumann: bool = True, 
    num_neumann_terms: int = 5
) -> torch.Tensor:
    """
    Perform the Cayley parametrization on a batch of parameter vectors Q.
    Args:
        Q: Parameter vectors of shape (b, n_elements)
    Returns:
        R: Orthogonal matrices of shape (b, block_size, block_size)
    """
    b, _ = Q.shape
    Q_skew = _pytorch_skew_symmetric(Q, block_size)

    if use_cayley_neumann:
        R = torch.eye(block_size, device=Q.device, dtype=Q.dtype).repeat(b, 1, 1)
        if num_neumann_terms > 1:
            R.add_(Q_skew, alpha=2.0)
            if num_neumann_terms > 2:
                Q_squared = torch.bmm(Q_skew, Q_skew)
                R.add_(Q_squared, alpha=2.0)
                Q_power = Q_squared
                for _ in range(3, num_neumann_terms - 1):
                    Q_power = torch.bmm(Q_power, Q_skew)
                    R.add_(Q_power, alpha=2.0)
                Q_power = torch.bmm(Q_power, Q_skew)
                R.add_(Q_power)
    else:
        id_mat = (
            torch.eye(block_size, device=Q.device, dtype=Q.dtype)
            .unsqueeze(0)
            .expand(b, block_size, block_size)
        )
        # Solve (I - Q) R = (I + Q)  => R = (I - Q)^-1 (I + Q)
        # Note: The official code calls it 'R' but returns the Orthogonal matrix usually denoted 'Q' or 'R' in OFT papers.
        # It solves (id_mat + Q_skew) X = (id_mat - Q_skew) -> X = (I+Q)^-1 (I-Q) ??
        # Wait, official code: torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew, left=False)
        # Solve AX = B.  Here A = (I + Q), B = (I - Q).
        # So it returns (I + Q)^-1 (I - Q). This is the Cayley transform.
        R = torch.linalg.solve(id_mat + Q_skew, id_mat - Q_skew)

    return R

def apply_oft_linear(
    x: torch.Tensor,
    oft_R_stacked: torch.Tensor, 
    adapter_idx: int = 0,
    block_share: bool = False
) -> torch.Tensor:
    """
    Applies Orthogonal Fine-Tuning (OFT) rotation to input x.
    
    x: (..., dim)
    oft_R_stacked: (max_ofts, 1, num_blocks, block_params)
    """
    # 1. Extract parameters
    # oft_R_stacked: (oft1, oft2, oft3, ...) (depending on the layer type / n_slices)
    # oft_R_stacked[adapter_idx]: (max_ofts, 1, num_blocks, block_params)

    # Handle merged layer (e.g., QKV) with optimized Cayley Batch
    if len(oft_R_stacked) > 1:
        # 1. Collect all parameters into a single batch
        # oft_R_stacked is a tuple of Tensors. Each tensor is (max_ofts, 1, num_blocks, params)
        # We assume adapter_idx=0 and single rank dim=0 for now. TODO: CHECK THIS!!!

        # Extract params for the active adapter
        params_list = [r[adapter_idx, 0] for r in oft_R_stacked]
        # Keep track of how many blocks each slice has (Q, K, V might have different sizes, e.g. GQA)
        num_blocks_list = [p.shape[0] for p in params_list]

        # Stack params into a single batch tensor
        all_params = torch.cat(params_list, dim=0)

        # 2. Determine Block Size
        block_params = all_params.shape[-1]
        block_size = int((1 + (1 + 8 * block_params)**0.5) / 2)

        # 3. Compute Rotation Matrices
        # orth_rotate_all: (total_blocks, block_size, block_size)
        orth_rotate_all = _cayley_batch(
            all_params,
            block_size,
        )

        # 4. Split back into Q, K, V orth rotation matrices
        orth_rotate_split = torch.split(orth_rotate_all, num_blocks_list, dim=0)

        # 5. Apply rotations to x
        rotated_inputs = []
        for i, orth_rotate in enumerate(orth_rotate_split):
            orig_shape = x.shape
            # Reshape x: (..., num_blocks, block_size)
            batch_dims = x.shape[:-1]
            x_reshaped = x.view(*batch_dims, num_blocks_list[i], block_size)
            # Applying rotation R to x. 
            x_rot = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)
            x_rot = x_rot.reshape(orig_shape) # Flatten back

            rotated_inputs.append(x_rot)

        return torch.cat(rotated_inputs, dim=0)

    params = oft_R_stacked[0][adapter_idx, 0] 
    num_blocks, block_params = params.shape
    # Calculate block_size: n*(n-1)/2 = block_params
    block_size = int((1 + (1 + 8 * block_params)**0.5) / 2)
    
    # 2. Compute Rotation Matrix R (block diagonal components)
    # orth_rotate: (num_blocks, block_size, block_size)
    orth_rotate = _cayley_batch(
        params, 
        block_size, 
    )
    
    # 3. Apply Rotation
    # x: (..., in_features)
    # We reshape x to (..., num_blocks, block_size)

    orig_shape = x.shape
    batch_dims = x.shape[:-1]
    x_reshaped = x.view(*batch_dims, num_blocks, block_size)
    x_rot = torch.einsum("...rk,rkc->...rc", x_reshaped, orth_rotate)
    
    return x_rot.reshape(orig_shape)