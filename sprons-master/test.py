import torch

def reconstruct_M_from_N(J):
    # Step 1: Create M from J (M = JJ^T)
    M = torch.mm(J, J.T)  # Original symmetric matrix M
    m = M.size(0)         # Size of M

    # Step 2: Create a smaller matrix N (n x n) from M
    n = int(m // 2)       # Define the size of N (for demonstration, use half the size of M)
    N = M[:n, :n].clone() # Extract top-left n x n submatrix as N

    # Step 3: Set selected elements of N to 0
    # For demonstration, zero out the diagonal of N
    for i in range(n):
        N[i, i] = 0

    # Step 4: Reconstruct M from the updated N
    # Create a new M_reconstructed with the original M
    M_reconstructed = M.clone()

    # Update the top-left n x n submatrix with the modified N
    M_reconstructed[:n, :n] = N

    # Return the reconstructed M
    return M, N, M_reconstructed

# Example usage
m = 6  # Size of the original matrix M
J = torch.randn(m, m)  # Generate a random matrix J

M, N, M_reconstructed = reconstruct_M_from_N(J)

print("Original Matrix M:")
print(M)
print("\nMatrix N (extracted and modified):")
print(N)
print("\nReconstructed Matrix M:")
print(M_reconstructed)
