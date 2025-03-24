using CUDA, LinearAlgebra, JLD2

GC.gc()    # Run Julia's garbge collector
CUDA.reclaim()  # Free unused GPU memory

# immutable struct to define key parameters
struct param       
    N::Int32        # number of 4-spin vertexes per side length ( e.g. N=2 for for a 4x4 system)
    L::Float32      # length of the in-plane elements
    L_c::Float32    # length of the verical control elements
    J::Float32      # NN distance ( set to 1 here)
    β::Float32      # angle defining geometry (i.e. angle spins make with x axis), 45 degrees for square ASI
    q::Float32      # pole strength for in-plane elements  (can be fractional)
    q_c::Float32    # pole strength for the control elements  (can be fractional)
    alpha::Float32  # rotation angle for pinwheel geometry (not used here)
end

 """
    get_control_positions(par::param, use_controls::Bool) -> (Vector{Float32}[], Int32)

Computes the positions of the vertical control elements (VCEs) if enabled.

# Arguments:
- `par::param`: The ASI system parameters.
- `use_controls::Bool`: Flag indicating whether control elements should be included.

# Returns:
- `control_positions::Vector{Float32}[]`: List of control element position vectors.
- `num_controls::Int32`: Number of control elements.
"""

function get_control_positions(par::param, use_controls::Bool)

    control_positions = Vector{Float32}[]

    # If controls are not enabled, return an empty array
    if use_controls == false
        return control_positions, 0
    end

    # x and y coordinates for placing VEs
    xx = collect(-Float32(par.N) : par.J : Float32(par.N))
    yy = collect(Float32(par.N) : - par.J : -Float32(par.N)) 

    # x- and y-position of all control spins (rightmost edge of the lattice)
    x_control = Int32(2 * par.N + 1)
    control_y_positions = yy[1:2:end]

    #Determine number of controls dynamically
    num_controls = length(control_y_positions)

    # creating VEs. If commented out, no VEs will be created.

    for y_pos in control_y_positions
        push!(control_positions, [xx[x_control], y_pos, 0.0f0]) 
    end
    
    return control_positions, num_controls
end

"""
    get_spin_centers(par::param, control_positions) -> Vector{Float32}[]

Generates the positions of spin centers in the ASI system, including control elements.

# Arguments:
- `par::param`: The ASI system parameters.
- `control_positions::Vector{Float32}[]`: List of control element positions.

# Returns:
- `spin_centers::Vector{Float32}[]`: List of all spin center positions in the system.
"""
function get_spin_centers(par::param, control_positions)

    # generate a periodic lattice of points

    # ASI array side length l in terms of J spacing
    # l is the distance from outermost spin centers on each side
    
    l = (2par.N - 1) * par.J

    # generate in-pane ASI elements' (x,y,z) position vectors
    spin_centers = Vector{Float32}[]
    
    for x = -l/2 : par.J : l/2 
        for y = l/2 : -par.J : -l/2
            push!(spin_centers, Float32[x,y,0.0f0])
        end
    end

    # generate control's (x,y,z) position vectors, control's pole sitting in the SASI plane
    for (i, v) in enumerate(control_positions)
        push!(spin_centers, [v[1], v[2], par.L_c/2.0f0]) 
    end

    return spin_centers
end

"""
    create_T2_config(par::param, control_positions) -> Vector{Float32}[]

Generates an initial random spin configuration for a single replica.

# Arguments:
- `par::param`: The ASI system parameters.
- `control_positions::Vector{Float32}[]`: List of control element positions.

# Returns:
- `μ_hat::Vector{Float32}[]`: A list of spin direction vectors for a system
consisting of T2 vertexes magnetized in the positive x-direction. 
"""

function create_T2_config(par::param, control_positions)

    # This function only helps to determne the left and right (x,y,z) charge positions, 
    # on either side of the spin center of each ASI element.
    
    # individual spin vectors grouped in "cells" of 4
    spin_1 = [cos(par.β), -sin(par.β), 0]
    spin_2 = [cos(par.β), cos(par.β), 0]
    spin_3, spin_4 = spin_2, spin_1

    # number of columns in the lattice (it is also a # of spins per side)
    columns = 2par.N

    μ_hat_T2 = Vector{Float32}[]

    for col = 1:columns
        spins = isodd(col) ? [spin_1, spin_2] : [spin_3, spin_4]
        append!(μ_hat_T2, repeat(spins, par.N))
    end

    # add control's spins 
     for i in eachindex(control_positions[1:length(control_positions)])
        push!(μ_hat_T2, [0.0f0, 0.0f0, par.L_c])       # z-component determines poles separation
                                                    # (+1 for UP spins! due to convention of L/R)
    end

    return μ_hat_T2
end


"""
    create_random_config(par::param, control_positions) -> Vector{Float32}[]

Generates an initial random spin configuration for a single replica.

# Arguments:
- `par::param`: The ASI system parameters.
- `control_positions::Vector{Float32}[]`: List of control element positions.

# Returns:
- `μ_hat::Vector{Float32}[]`: A list of spin direction vectors representing the initial state incl. control elements.
"""
function create_random_config(par::param, control_positions)

    # individual spin vectors grouped in "cells" or "vertexes" of 4 spins in T2 config
    spin_1 = [cos(par.β), -sin(par.β), 0]
    spin_2 = [cos(par.β), cos(par.β), 0]
    spin_3, spin_4 = spin_2, spin_1

    # number of columns in the lattice (it is also a # of spins per side)
    columns = 2par.N
    
    μ_hat = Vector{Float32}[]

    # in-plane spins
    for col = 1:columns
        spins = isodd(col) ? [spin_1 *rand([1,-1]) , spin_2 *rand([1,-1])] : [spin_3 *rand([1,-1]), spin_4 *rand([1,-1])]
        append!(μ_hat, repeat(spins, par.N))
    end

    # control spins
    for i in eachindex(control_positions[1:length(control_positions)])
        push!(μ_hat, [0.0f0, 0.0f0, par.L_c])          # z-component determines poles separation
                                                    # (+1 for UP spins! due to convention of L/R)
    end

    return μ_hat
end

"""
    get_charge_positions(par::param, control_positions, spin_centers) -> (Vector{Float32}[], Vector{Float32}[])

Computes the positions of magnetic charges to the left of the spin centers and to the right in the artificial spin ice system.

# Arguments:
- `par::param`: The ASI system parameters.
- `control_positions::Vector{Float32}[]`: List of control element positions.
- `spin_centers::Vector{Float32}[]`: List of all spin center positions.

# Returns:
- `positions_left::Vector{Float32}[]`: List of positions for the left charge at each spin.
- `positions_right::Vector{Float32}[]`: List of positions for the right charge at each spin.

# Notes:
- This function must be run on a **T2 positively magnetized system configuration**.
- If an arbitrary random configuration is used, the charge positioning convention will be incorrect.
- The computed positions are critical for ensuring correct dipolar interaction calculations.
- No assumption about the polarity of the charge is made.
"""

function get_charge_positions(par::param, control_positions, spin_centers)

    # individual spin vectors grouped in "cells" of 4
    spin_1 = [cos(par.β), -sin(par.β), 0]
    spin_2 = [cos(par.β), cos(par.β), 0]
    spin_3, spin_4 = spin_2, spin_1

    # number of columns in the lattice (it is also a # of spins per side)
    columns = 2par.N

    μ_hat_T2 = Vector{Float32}[]

    for col = 1:columns
        spins = isodd(col) ? [spin_1, spin_2] : [spin_3, spin_4]
        append!(μ_hat_T2, repeat(spins, par.N))
    end

    # add control's spins 
    for i in eachindex(control_positions[1:length(control_positions)])
        push!(μ_hat_T2, [0.0f0, 0.0f0, par.L_c])      
    end                                               


    positions_left = Vector{Float32}[]
    positions_right = Vector{Float32}[]

    for i in eachindex(spin_centers, μ_hat_T2)
        push!(positions_left, spin_centers[i] - 0.5f0 * par.L * μ_hat_T2[i])
        push!(positions_right, spin_centers[i] + 0.5f0 * par.L * μ_hat_T2[i])
    end

    return positions_left, positions_right
end


"""
    precompute_neighbor_list(spin_centers, R_c, num_spins) -> Vector{Vector{Int32}}

Precomputes the list of neighboring spins for each spin based on a cutoff radius.

# Arguments:
- `spin_centers::Vector{Float32}[]`: List of spin center positions.
- `R_c::Float32`: Cutoff radius for considering spins as neighbors.
- `num_spins::Int32`: Total number of spins in the system.

# Returns:
- `neighbor_list::Vector{Vector{Int32}}`: A list where each entry contains the indices of neighboring spins within the cutoff radius.

# Notes:
- Runs on the **CPU** before the main simulation to avoid recomputing neighbors during Monte Carlo steps.
- Uses Euclidean distance to determine neighbor relationships.
- The cutoff radius is optimized to balance computational efficiency and accuracy.
"""

function precompute_neighbor_list(spin_centers, R_c, num_spins)
    
    neighbor_list = Vector{Vector{Int32}}(undef, num_spins)

    for i in 1:num_spins
        neighbors = Int32[]
        for j in 1:num_spins
            if i != j && norm(spin_centers[i] .- spin_centers[j]) <= R_c
                push!(neighbors, j)
            end
        end
        neighbor_list[i] = neighbors
    end

    return neighbor_list
end


"""
    convert_neighbor_list_to_matrix(neighbor_list, max_neighbors, num_spins) -> CuArray{Int32, 2}

Converts a variable-length neighbor list into a fixed-size matrix for efficient GPU access.

# Arguments:
- `neighbor_list::Vector{Vector{Int32}}`: List where each entry contains indices of neighboring spins.
- `max_neighbors::Int32`: Maximum number of neighbors any spin can have.
- `num_spins::Int32`: Total number of spins in the system.

# Returns:
- `neighbor_matrix::CuArray{Int32, 2}`: A `num_spins × max_neighbors` CUDA array where each row contains neighbor indices, padded with `-1` for invalid entries.

# Notes:
- This function ensures a **fixed memory layout** suitable for efficient GPU kernel access.
- Any missing neighbor slots are filled with `-1` to indicate invalid indices.
- The output is moved to GPU memory (`CuArray`) for fast access in CUDA kernels.
"""
function convert_neighbor_list_to_matrix(neighbor_list, max_neighbors, num_spins)
    
    neighbor_matrix = fill(Int32(-1), num_spins, max_neighbors)  # Fill with -1 (invalid index)

    for i in 1:num_spins
        for (k, j) in enumerate(neighbor_list[i])
            if k <= max_neighbors
                neighbor_matrix[i, k] = j  # Store valid neighbor index
            end
        end
    end
    return cu(neighbor_matrix)
end



"""
    precompute_J_matrix_same_side(position_vectors, control_positions, neighbor_list, use_controls) -> CuArray{Float32, 2}

Precomputes the dipolar interaction matrix for magnetic poles/charges on the **same side** relative to the spin center in the dumbbell model.

# Arguments:
- `position_vectors::Vector{Float32}[]`: List of spin center positions.
- `control_positions::Vector{Float32}[]`: List of control element positions.
- `neighbor_list::Vector{Vector{Int32}}`: Precomputed list of neighboring spins.
- `use_controls::Bool`: Flag to include control elements in interaction calculations.

# Returns:
- `J_compressed::CuArray{Float32, 2}`: A `num_spins × (max_neighbors + num_controls)` CUDA array containing interaction strengths.

# Notes:
- The interaction strength between two charges is computed as **1 / distance**.
- If `use_controls` is `true`, interactions between spins and control elements are also included.
- The matrix is **compressed** by storing only valid neighbors (instead of a full dense matrix), reducing memory usage and improving computation efficiency.
- The output is stored as a `CuArray` for **fast GPU access** in CUDA kernels.
"""

function precompute_J_matrix_same_side(position_vectors, control_positions, neighbor_list, use_controls)
    
    num_spins = length(position_vectors)
    num_neighbors = maximum(length(n) for n in neighbor_list)  # Max neighbors any spin has
    num_controls = length(control_positions)  # Number of control spins
  
    # Preallocate with extra columns for control interactions
    J_compressed = fill(0.0f0, num_spins, num_neighbors + num_controls) 

    for i in 1:num_spins
        # Regular neighbors (within R_c)
        for (k, j) in enumerate(neighbor_list[i])  # Only store valid neighbors
            J_compressed[i, k] = 1 / norm(position_vectors[i] .- position_vectors[j])
        end

        # Control interactions (every spin interacts with ALL controls)
        if use_controls
            offset = num_neighbors  # Start storing controls after regular neighbors
            for (c_idx, c_pos) in enumerate(control_positions)
                J_compressed[i, offset + c_idx] = 1 / norm(position_vectors[i] .- c_pos)
            end
        end
    end

    return cu(J_compressed)
end

"""
    precompute_J_matrix_opposite_side(position_vectors1, position_vectors2, control_positions, neighbor_list, use_controls) -> CuArray{Float32, 2}

Precomputes the dipolar interaction matrix for magnetic poles/charges on the **opposite sides** relative to the spin center in the dumbbell model.

# Arguments:
- `position_vectors::Vector{Float32}[]`: List of spin center positions.
- `control_positions::Vector{Float32}[]`: List of control element positions.
- `neighbor_list::Vector{Vector{Int32}}`: Precomputed list of neighboring spins.
- `use_controls::Bool`: Flag to include control elements in interaction calculations.

# Returns:
- `J_compressed::CuArray{Float32, 2}`: A `num_spins × (max_neighbors + num_controls)` CUDA array containing interaction strengths.

# Notes:
- The interaction strength between two charges is computed as **1 / distance**.
- If `use_controls` is `true`, interactions between spins and control elements are also included.
- The matrix is **compressed** by storing only valid neighbors (instead of a full dense matrix), reducing memory usage and improving computation efficiency.
- The output is stored as a `CuArray` for **fast GPU access** in CUDA kernels.
"""
function precompute_J_matrix_opposite_side(position_vectors_1, position_vectors_2, control_positions, neighbor_list, use_controls)
    
    num_spins = length(position_vectors_1)
    num_neighbors = maximum(length(n) for n in neighbor_list)  # Max neighbors any spin has
    num_controls = length(control_positions)  # Number of control spins
  
    # Preallocate with extra columns for control interactions
    J_compressed = fill(0.0f0, num_spins, num_neighbors + num_controls) 

    for i in 1:num_spins
        # Regular neighbors (within R_c)
        for (k, j) in enumerate(neighbor_list[i])  # Only store valid neighbors
            J_compressed[i, k] = 1 / norm(position_vectors_1[i] .- position_vectors_2[j])
        end

        # Control interactions (every spin interacts with ALL controls)
        if use_controls
            offset = num_neighbors  # Start storing controls after regular neighbors
            for (c_idx, c_pos) in enumerate(control_positions)
                J_compressed[i, offset + c_idx] = 1 / norm(position_vectors_1[i] .- c_pos)
            end
        end
    end

    return cu(J_compressed)
end

"""
    create_multiple_replicas(par::param, control_positions, num_plane_spins, num_spins, num_replicas) -> (CuArray{Float32, 2}, CuArray{Float32, 2}, CuArray{Float32, 2})

Generates multiple random initial spin configurations for Monte Carlo replicas.

# Arguments:
- `par::param`: ASI system parameters.
- `control_positions::Vector{Float32}[]`: List of control element positions.
- `num_plane_spins::Int32`: Number of spins in the plane of ASI lattice.
- `num_spins::Int32`: Total number of spins, including in-plane and control spins.
- `num_replicas::Int32`: Number of independent replicas to generate.

# Returns:
- `S::CuArray{Float32, 2}`: Matrix of spin states (`+1` or `-1`), indicating left/right orientation.
- `SX::CuArray{Float32, 2}`: Matrix storing the x-components of spin vectors for each replica.
- `SY::CuArray{Float32, 2}`: Matrix storing the y-components of spin vectors for each replica.

# Notes:
- **Runs on the CPU** and generates **random initial states** for all replicas.
- The `SpinStates` matrix is used for charge product calculations in the dumbbell energy model.
- The `SX` and `SY` matrices store spin vector components for magnetization and field energy computations.
- Control spin states are handled separately from in-plane spins using `control_q`.
- The resulting arrays are converted to GPU arrays (`CuArray`) for efficient processing in CUDA kernels.
"""
function create_multiple_replicas(par::param, control_positions, num_plane_spins, num_spins, num_replicas)
    
    SpinStates = zeros(Float32, num_spins, num_replicas)
    SX = zeros(Float32, num_spins, num_replicas)
    SY = zeros(Float32, num_spins, num_replicas)

    for r = 1:num_replicas
        # Generate a random initial state
        μ_hat = create_random_config(par, control_positions)
        Mu = Matrix{Float32}(reduce(vcat,transpose.(μ_hat))) # convert vector of vectors to a (num_spins x 3) matrix
    
        # Generate T2 initial state if needed
        # μ_hat_T2 = create_T2_config(par, vertex_positions)
        # Mu = Matrix{Float32}(reduce(vcat,transpose.(μ_hat_T2))) # convert vector of vectors to a (num_spins x 3) matrix

        # Store spin vector components
        SX[:, r] = Mu[:,1]
        SY[:, r] = Mu[:,2]

        # Compute spin state based on x-component sign (left/right orientation)
        plane_q = sign.(Mu[1:num_plane_spins, 1])               
        control_q = Mu[num_plane_spins + 1:end, 3] .* par.q_c   # Control spins (UP state)

        # Final spin state matrix (used for dumbbell energy calculations)
        SpinStates[:, r] = cat(plane_q, control_q; dims=1) 
    end

    # Convert CPU arrays to CUDA arrays for GPU computation
    S = SpinStates |> cu
    SX = SX |> cu
    SY = SY |> cu

    return S, SX, SY
end

"""
    compute_dumbell_energy_kernel!(...)

CUDA kernel function to compute dumbbell energy contributions for each spin in the system.

# Arguments:
- `J_LL, J_RR, J_LR, J_RL`: Precomputed interaction matrices.
- `S`: Spin state matrix (num_spins × num_replicas).
- `neighbor_matrix`: List of neighbors for each spin, matrix of size (num_spins, max_neighbors).
- `dumbell_energies`: Array (1, num_replicas) to store computed energies.
- `num_spins, num_replicas, max_neighbors`: System parameters.

# Notes:
- Uses atomic addition to accumulate energy contributions.
- Parallelized across spins and replicas.
"""
function compute_dumbell_energy_kernel!(
    J_LL, J_RR, J_LR, J_RL, S, neighbor_matrix,
    dumbell_energies, num_spins, num_replicas, max_neighbors
)
    ### Compute Thread & Block Indexing
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x  # Spin index (row in S)
    r = blockIdx().y  # Replica index (column in S)

    ### Bounds Check
    if i <= num_spins && r <= num_replicas
        local_energy = 0.0f0

        ### Iterate Over Neighbors
        for k in 1:max_neighbors
            j = neighbor_matrix[i, k]  # Get neighbor spin index

            # Ignore invalid (-1) neighbor entries
            if j != -1 && i < j  # Prevent double-counting by ensuring i < j
                local_energy += J_LL[i, k] * -S[i, r] * -S[j, r]
                local_energy += J_RR[i, k] * S[i, r] * S[j, r]
                local_energy += J_LR[i, k] * -S[i, r] * S[j, r]
                local_energy += J_RL[i, k] * S[i, r] * -S[j, r]
            end
        end

        ### Accumulate Energy Using Atomic Addition
        CUDA.atomic_add!(CUDA.pointer(dumbell_energies, r), local_energy)
    end

    return
end

"""
    compute_magnetization_kernel!(SX, SY, unit_hx, unit_hy, num_plane_spins, num_replicas, magnetizations)

CUDA kernel that computes the projected magnetization per replica using a parallel reduction algorithm with shared memory.

# Arguments:
- `SX::CuArray{Float32, 2}`: X-components of spin vectors (num_spins × num_replicas).
- `SY::CuArray{Float32, 2}`: Y-components of spin vectors (num_spins × num_replicas).
- `unit_hx::Float32`: X-component of the unit vector along the applied field direction.
- `unit_hy::Float32`: Y-component of the unit vector along the applied field direction.
- `num_plane_spins::Int32`: Number of spins in the in-plane ASI lattice.
- `num_replicas::Int32`: Total number of replicas being simulated.
- `magnetizations::CuArray{Float32, 1}`: Output array storing the computed magnetization for each replica.

# CUDA Parallelization Strategy:
- Each replica is assigned a column of blocks.
- Each thread processes a single spin per replica.
- Multiple blocks per replica are used to distribute large spin arrays across threads.

# CUDA Grid Configuration:
- `blockIdx().x` → Identifies the **replica** (each replica gets its own column of blocks).
- `blockIdx().y` → Multiple **blocks per replica** (to handle large spin lattices).
- `threadIdx().y` → Each **thread processes one spin** in the assigned replica.

# Algorithm Details:
1. **Each thread computes a local contribution** to the projected magnetization from its assigned spin.
2. **Shared memory reduction**: Each block accumulates partial sums using a parallel reduction algorithm.
   - Shared memory (`shared_M`) is used to store partial sums.
   - Reduction occurs in **logarithmic steps** (`stride /= 2` at each iteration).
3. **Final global accumulation**: Only one thread per block writes the reduced sum to `magnetizations` using **atomic operations**.

# Notes:
- Atomic operations (`CUDA.atomic_add!`) ensure safe accumulation of partial results from multiple blocks.
- Shared memory usage improves efficiency by avoiding frequent global memory access.
- Optimized for large-scale simulations where each replica has thousands of spins.
"""
function compute_magnetization_kernel!(SX, SY, unit_hx, unit_hy, num_plane_spins, num_replicas, magnetizations)
  
    # Thread & Block Identifiers
    thread_id = threadIdx().y  # Each thread handles one spin per replica
    block_id = blockIdx().y  # Block index within a replica (allows multiple blocks per replica)
    replica_id = blockIdx().x  # Each replica is handled independently

    # Compute Spin Index This Thread is Responsible For
    spins_per_block = blockDim().y  # Number of threads (spins) per block
    spin_start = (block_id - 1) * spins_per_block + thread_id  # Julia is **one-based indexing**

    # Shared Memory for Partial Sum Reduction
    shared_M = CuDynamicSharedArray(Float32, spins_per_block)

    # Initialize Local Sum
    M_local = 0.0f0

    # Bounds Check: Only Process Valid Spins
    if spin_start <= num_plane_spins
        M_local += unit_hx * SX[spin_start, replica_id] + unit_hy * SY[spin_start, replica_id]
    end

    # Store Local Sum in Shared Memory
    shared_M[thread_id] = M_local
    sync_threads()

    #*Parallel Reduction Within Each Block
    stride = spins_per_block ÷ 2  # Start reduction at half the block size
    while stride > 0
        if thread_id <= stride && (thread_id + stride) <= spins_per_block
            shared_M[thread_id] += shared_M[thread_id + stride]
        end
        stride ÷= 2
        sync_threads()
    end

    # Global Accumulation (Only One Thread Writes)
    if thread_id == 1  # First thread in each block writes its sum to global memory
        CUDA.atomic_add!(CUDA.pointer(magnetizations, replica_id), shared_M[1]) 
    end

    return
end

"""
    compute_field_energy_kernel!(SX, SY, h, unit_hx, unit_hy, num_plane_spins, num_replicas, field_energies)

CUDA kernel that computes the field energy per replica using a parallel reduction algorithm with shared memory.

# Arguments:
- `SX::CuArray{Float32, 2}`: X-components of spin vectors (num_spins × num_replicas).
- `SY::CuArray{Float32, 2}`: Y-components of spin vectors (num_spins × num_replicas).
- `h::Float32`: Magnitude (incl. sign) of the applied external field.
- `unit_hx::Float32`: X-component of the unit vector along the field direction.
- `unit_hy::Float32`: Y-component of the unit vector along the field direction.
- `num_plane_spins::Int32`: Number of spins in the in-plane ASI lattice.
- `num_replicas::Int32`: Total number of replicas being simulated.
- `field_energies::CuArray{Float32, 1}`: Output array storing the computed field energy for each replica.

# CUDA Parallelization Strategy:
- Each replica is assigned a column of blocks.
- Each thread processes a single spin per replica.
- Multiple blocks per replica are used to distribute large spin arrays across threads.

# CUDA Grid Configuration:
- `blockIdx().x` → Identifies the **replica** (each replica gets its own column of blocks).
- `blockIdx().y` → Multiple **blocks per replica** (to handle large spin lattices).
- `threadIdx().y` → Each **thread processes one spin** in the assigned replica.

# Algorithm Details:
1. **Each thread computes a local contribution** to the field energy from its assigned spin.
2. **Shared memory reduction**: Each block accumulates partial sums using a parallel reduction algorithm.
   - Shared memory (`shared_Eh`) is used to store partial sums.
   - Reduction occurs in **logarithmic steps** (`stride /= 2` at each iteration).
3. **Final global accumulation**: Only one thread per block writes the reduced sum to `field_energies` using **atomic operations**.

# Notes:
- Atomic operations (`CUDA.atomic_add!`) ensure safe accumulation of partial results from multiple blocks.
- Shared memory usage improves efficiency by avoiding frequent global memory access.
- Optimized for large-scale simulations where each replica has thousands of spins.
"""

function compute_field_energy_kernel!(SX, SY, h, unit_hx, unit_hy, num_plane_spins, num_replicas, field_energies)
 
    # Thread & Block Identifiers
    thread_id = threadIdx().y  # Each thread handles one spin per replica
    block_id = blockIdx().y  # Block index within a replica (for multiple blocks per replica)
    replica_id = blockIdx().x  # Each replica is handled independently

    # Compute Spin Index This Thread is Responsible For
    spins_per_block = blockDim().y  # Number of threads (spins) per block
    spin_start = (block_id - 1) * spins_per_block + thread_id  # Julia is **one-based indexing**

    # Shared Memory for Partial Sum Reduction
    shared_Eh = CuDynamicSharedArray(Float32, spins_per_block)

    # Initialize Local Sum
    E_local = 0.0f0

    # Bounds Check: Only Process Valid Spins
    if spin_start <= num_plane_spins
        E_local += h * (unit_hx * SX[spin_start, replica_id] + unit_hy * SY[spin_start, replica_id])
    end

    # Store Local Sum in Shared Memory
    shared_Eh[thread_id] = E_local
    sync_threads()

    # Parallel Reduction Within Each Block
    stride = spins_per_block ÷ 2  # Start reduction at half the block size
    while stride > 0
        if thread_id <= stride && (thread_id + stride) <= spins_per_block
            shared_Eh[thread_id] += shared_Eh[thread_id + stride]
        end
        stride ÷= 2
        sync_threads()
    end

    # Global Accumulation (Only One Thread Writes)
    if thread_id == 1  # First thread in each block writes its sum to global memory
        CUDA.atomic_add!(CUDA.pointer(field_energies, replica_id), shared_Eh[1]) 
    end
    
    return
end

"""
    metropolis_step_kernel!(
        J_LL, J_RR, J_LR, J_RL, S, SX, SY, neighbor_matrix, num_spins, num_plane_spins,
        dumbell_energies, rand_indices, metro_rand_vals, 
        h, unit_hx, unit_hy, T, num_replicas, flip_attempt, max_neighbors
    )

CUDA kernel implementing the Metropolis Monte Carlo spin-flip update for the ASI system.

# Arguments:
- `rand_indices::CuArray{Int32, 2}`: Precomputed random spin selection indices.
- `metro_rand_vals::CuArray{Float32, 2}`: Precomputed random numbers for the Metropolis criterion.
- `h::Float32`: Applied external field magnitude incl. sign.
- `unit_hx, unit_hy::Float32`: Components of the unit field direction vector.
- `T::Float32`: Simulation temperature (in reduced units).
- `flip_attempt::Int32`: Index of the spin flip attempt in the Monte Carlo sequence.
- `max_neighbors::Int32`: Maximum number of neighbors any spin has.

# CUDA Parallelization Strategy:
- Each replica is assigned to a block (`blockIdx().x`).
- Each thread (`threadIdx().x`) computes the energy contribution of a single neighbor interaction.

# Algorithm Details:
1. **Random Spin Selection:** Each replica selects a random spin `i` to attempt flipping.
2. **Neighbor Contributions:**
   - Each thread processes one neighbor `j` of `i` and computes its contribution to the total energy change.
   - The dipolar energy is calculated **before and after** the spin flip.
   - The **shared memory array** (`shared_deltaE_db`) stores the energy differences for all neighbors for this replica.
3. **Shared Memory Reduction:**
   - A reduction step sums all **thread-local energy contributions** in **logarithmic** time.
4. **Field Energy Calculation:**
   - The contribution of the **applied external field** to the total energy is computed separately.
5. **Metropolis Acceptance Criterion:**
   - If the total energy change `ΔE_total` is **negative**, the flip is **accepted**.
   - If `ΔE_total > 0`, the flip is accepted **probabilistically** using `exp(-ΔE_total / T)`.
6. **Atomic Updates:**
   - If the spin flip is accepted:
     - The **global dumbbell energy** is updated using `CUDA.atomic_add!`.
     - The spin state (`S`), `SX`, and `SY` matrices are **updated in-place**.

# Notes:
- Atomic operations ensure energy updates are correctly accumulated across threads.
- Precomputed random indices and values improve performance by avoiding runtime random number generation.
- Optimized for large-scale simulations, efficiently handling thousands of spins per replica.

# Performance:
- Shared memory usage reduces global memory access latency.
"""
function metropolis_step_kernel!(
    J_LL, J_RR, J_LR, J_RL, S, SX, SY, neighbor_matrix, num_spins, num_plane_spins,
    dumbell_energies, rand_indices, metro_rand_vals, 
    h, unit_hx, unit_hy, T, num_replicas, flip_attempt, max_neighbors
)
    """
    CUDA Grid:
    - blockIdx().x → Each replica (R) gets a block.
    - threadIdx().x → Each thread in a block computes i-j energy contribution.
    """

    # Thread and Block Identifiers
    r = blockIdx().x  # Each block handles one replica
    k = threadIdx().x  # Each thread processes one neighbor interaction

    # Shared Memory for Partial Sums
    shared_deltaE_db = CuDynamicSharedArray(Float32, blockDim().x)

    # Bounds Check
    if r <= num_replicas && flip_attempt <= num_plane_spins

        ### Select the Spin to Flip in This Replica
        i = rand_indices[flip_attempt, r]  # Index of the spin to flip

        if i <= num_plane_spins

            ### Ensure the thread is within bounds of neighbors
            if k <= max_neighbors
                j = neighbor_matrix[i, k]  # Retrieve neighbor index

                if j != -1  && j <= num_spins # Ensure valid neighbor
                    ### Compute Interaction Energy Before and After Flip
                    Si_old = S[i, r]  # Old spin value
                    Sj = S[j, r]  # Neighbor spin value

                    Si_new = -Si_old  # Temporarily flipped spin

                    ### Compute Energy Contribution Before Flip
                    E_old = 0.0f0
                    E_old += J_LL[i, k] * (-Si_old * -Sj)
                    E_old += J_RR[i, k] * (Si_old * Sj)
                    E_old += J_LR[i, k] * (-Si_old * Sj)
                    E_old += J_RL[i, k] * (Si_old * -Sj)

                    ### Compute Energy Contribution After Flip
                    E_new = 0.0f0
                    E_new += J_LL[i, k] * (-Si_new * -Sj)
                    E_new += J_RR[i, k] * (Si_new * Sj)
                    E_new += J_LR[i, k] * (-Si_new * Sj)
                    E_new += J_RL[i, k] * (Si_new * -Sj)

                    ### Store Energy Difference Per Thread in Shared Memory
                    shared_deltaE_db[k] = E_new - E_old
                else
                    shared_deltaE_db[k] = 0.0f0
                end

                sync_threads()

                ### Reduce Partial Energy Contributions
                if k == 1  # Use only the first thread to sum up the shared memory
                    deltaE_db = 0.0f0
                    for idx in 1:max_neighbors
                        deltaE_db += shared_deltaE_db[idx]
                    end
                    sync_threads()

                    ### Compute Field Energy Change (Single Thread)
                    Sx_old = SX[i, r]
                    Sy_old = SY[i, r]
                    Eh_old = 0.0f0
                    Eh_old = h * (unit_hx * Sx_old + unit_hy * Sy_old)

                    Sx_new = -Sx_old
                    Sy_new = -Sy_old
                    Eh_new = 0.0f0
                    Eh_new = h * (unit_hx * Sx_new + unit_hy * Sy_new)

                    deltaE_h = 0.0f0
                    deltaE_h = Eh_new - Eh_old
                    sync_threads()

                    ### Compute Total ΔE and Apply Metropolis Criterion
                    deltaE_total = 0.0f0
                    deltaE_total = deltaE_db - deltaE_h
                    accept_flip = (deltaE_total <= 0.0f0) || (exp(-deltaE_total / T) >= metro_rand_vals[flip_attempt, r])
                
                    if accept_flip
                        # **Accept Flip: Update Global Energies**
                        CUDA.atomic_add!(CUDA.pointer(dumbell_energies, r), deltaE_db)
                    
                        # Flip the Spin in S, SX, and SY permanently
                        S[i, r] *= -1
                        SX[i, r] *= -1
                        SY[i, r] *= -1
                    end
                end
            end
        end
    end

    return
end

"""
    metropolis_step!(
        J_LL, J_RR, J_LR, J_RL, S, SX, SY, neighbor_matrix,
        dumbell_energies, field_energies, magnetizations, rand_indices, metro_rand_vals,
        h, unit_hx, unit_hy, T, num_replicas, num_spins, num_plane_spins, max_neighbors
    )

Performs multiple Monte Carlo spin-flip attempts per replica using the Metropolis algorithm.

# CUDA Parallelization Strategy:
- **Each replica is assigned a block (`blockIdx().x`).**
- **Each thread (`threadIdx().x`) processes a neighbor interaction.**

# Algorithm Details:
1. **Monte Carlo Loop:**
   - Loops over all spins in the system, attempting **one spin flip per iteration**.
2. **Launch `metropolis_step_kernel!`:**
   - Computes **energy change (`ΔE`)** from neighbor interactions and applied field.
   - Applies the **Metropolis acceptance criterion**.
   - Updates spin states and global dumbell energies if the flip is accepted.
3. **Synchronize CUDA threads** (`CUDA.synchronize()`). Also helps to catch any kernel errors.
4. **Compute System Magnetization:**
   - Resets `magnetizations` to avoid accumulation from previous steps.
   - Launches `compute_magnetization_kernel!` to sum the total system magnetization.
5. *(Optional)* **Compute Field Energy:**
   - If needed, launches `compute_field_energy_kernel!` to track energy changes due to the external field.

# Notes:
- Precomputed random indices and values improve efficiency by avoiding runtime random number generation.

# Performance:
- Shared memory usage reduces global memory latency.
- CUDA grid configuration is optimized for large-scale ASI simulations.
- Supports thousands of spins and hundreds of replicas in parallel.
"""
function metropolis_step!(J_LL, J_RR, J_LR, J_RL, S, SX, SY, neighbor_matrix, 
    dumbell_energies, field_energies, magnetizations, rand_indices, metro_rand_vals, 
    h, unit_hx, unit_hy, T, num_replicas, num_spins, num_plane_spins, max_neighbors)
    """
    Performs multiple spin flip attempts per replica using Metropolis criterion.
    """

    block_size_metro = max_neighbors  # Use one thread per i-j interaction
    grid_size_metro = num_replicas  # One block per replica
    
    for flip_attempt = 1:num_plane_spins
        # Kernel for Metropolis updates (performs spin flips & energy calculations)
        @cuda blocks=grid_size_metro threads=block_size_metro shmem=(max_neighbors * sizeof(Float32)) metropolis_step_kernel!(
            J_LL, J_RR, J_LR, J_RL, S, SX, SY, neighbor_matrix, num_spins, num_plane_spins,
            dumbell_energies, rand_indices, metro_rand_vals, 
            h, unit_hx, unit_hy, T, num_replicas, flip_attempt, max_neighbors
        )
    end
   CUDA.synchronize()

    threads_per_block_m = Int32(256)  # Max threads per block
    blocks_per_replica_m = Int32(cld(num_plane_spins, threads_per_block_m))  # Multiple blocks per replica
    grid_size_m = (Int32(num_replicas), Int32(blocks_per_replica_m))  # 2D grid
    shared_mem_size_m = sizeof(Float32) * threads_per_block_m  # Shared memory per block
    
    CUDA.fill!(magnetizations, 0.0f0) # need this step to prevent overaccumulation of M

    # Launch Kernel
    @cuda threads=(Int32(1), threads_per_block_m) blocks=grid_size_m shmem=shared_mem_size_m compute_magnetization_kernel!(
        SX, SY, unit_hx, unit_hy, num_plane_spins, num_replicas, magnetizations
    )

    # this computation is not needed unless we explicitly want to track the field energy.

    # threads_per_block_h = Int32(256)  # Max threads per block
    # blocks_per_replica_h = Int32(cld(num_plane_spins, threads_per_block_h))  # Multiple blocks per replica
    # grid_size_h = (Int32(num_replicas), Int32(blocks_per_replica_h))  # 2D grid
    # shared_mem_size_h = sizeof(Float32) * threads_per_block_h  # Shared memory per block

    # CUDA.fill!(field_energies, 0.0f0) # need this step to prevent overaccumulation of E_field

    # # Launch Kernel
    # @cuda threads=(Int32(1), threads_per_block_h) blocks=grid_size_h shmem=shared_mem_size_h compute_field_energy_kernel!(
    #     SX, SY, h, unit_hx, unit_hy, num_plane_spins, num_replicas, field_energies
    # )

    return
end



###############################   MAIN function  ##############################

function simulate_system_hysteresis(num_plane_spins::Int32, num_replicas::Int32, use_controls::Bool, mcSteps::Int32)

    ### Simulation Parameters ###

    hstep = 0.19f0        # Adjust step size if needed
    H_max = 15.0f0        # Maximum applied field
    T = 0.3f0             # Set temperature
    q = 1.0f0             # Pole strength value
    q_c = 1.0f0           # Control pole strength value
    R_c = 6.0f0           # Radius of convergence for min neighbors needed to keep energy error < 1%
    N = Int32(sqrt(num_plane_spins)/2)  # Vertexes per sidelength (System size N = 1 gives (1x1 vertex or 2x2 = 4 spins))
        
    theta_deg = 0.1f0     # Symmetry-breakinf field angle in degrees
    theta_rad = Float32(theta_deg * π / 180)  # Convert to radians
    unit_hx, unit_hy = cos(theta_rad), sin(theta_rad) # Field direction unit vectors precomputed

    par = param(N, 1.0f0, 1.0f0, 1.0, π / 4.0f0, q, q_c, 0.0f0)

    #### Initialize the system and move the arrays to GPU ####

    control_positions, num_controls = get_control_positions(par, use_controls)
    num_spins = num_plane_spins + num_controls |> Int32
    spin_centers = get_spin_centers(par, control_positions)
	
	println("Number of controls is ", num_controls )
	println("Total number of spins is ", num_spins )

    neighbor_list = precompute_neighbor_list(spin_centers, R_c, num_spins)
    max_neighbors = maximum(length(n) for n in neighbor_list) |> Int32
    neighbor_matrix = convert_neighbor_list_to_matrix(neighbor_list, max_neighbors, num_spins)

    positions_left, positions_right = get_charge_positions(par, control_positions, spin_centers)
    J_LL = precompute_J_matrix_same_side(positions_left, control_positions, neighbor_list, use_controls)
    J_RR = precompute_J_matrix_same_side(positions_right, control_positions, neighbor_list, use_controls)
    J_LR = precompute_J_matrix_opposite_side(positions_left, positions_right, control_positions, neighbor_list, use_controls)
    J_RL = precompute_J_matrix_opposite_side(positions_right, positions_left, control_positions, neighbor_list, use_controls)

    S, SX, SY = create_multiple_replicas(par::param, control_positions, num_plane_spins, num_spins, num_replicas)

    println("Running simulation with $(num_plane_spins) per replica, and $(num_replicas) replicas, and $(mcSteps) MC steps per field value")
    println("Dimensions of S, SX, SY matrix are ", size(S))
    println("Maximum number of in-plane neigbours for each spin is ", max_neighbors)
    println("Dimensions of J matrices are ", size(J_LL))

    #### Preallocate GPU Buffers in global memory (Allocated Once)  ####

    dumbell_energies = CUDA.zeros(Float32, num_replicas)
    field_energies = CUDA.zeros(Float32, num_replicas)
    magnetizations = CUDA.zeros(Float32, num_replicas)


    #### Compute Initial Energies and Magnetizations (Only Once)  ####

    block_size_db = min(Int32(256), num_spins)  # Tune for best performance
    grid_size_x = ceil(Int32, num_spins / block_size_db)  # Spins per grid
    grid_size_y = num_replicas  # Replicas per grid

    # **Launch Kernel**
    @cuda blocks=(grid_size_x, grid_size_y) threads=block_size_db compute_dumbell_energy_kernel!(
        J_LL, J_RR, J_LR, J_RL, S, neighbor_matrix,
        dumbell_energies, num_spins, num_replicas, max_neighbors
    )

    
    threads_per_block_h = Int32(256)  # Max threads per block
    blocks_per_replica_h = Int32(cld(num_plane_spins, threads_per_block_h))  # Multiple blocks per replica
    grid_size_h = (Int32(num_replicas), Int32(blocks_per_replica_h))  # 2D grid
    shared_mem_size_h = sizeof(Float32) * threads_per_block_h  # Shared memory per block
    h_i = 0.0f0 # at the start

    # **Launch Kernel**
    @cuda threads=(Int32(1), threads_per_block_h) blocks=grid_size_h shmem=shared_mem_size_h compute_field_energy_kernel!(
        SX, SY, h_i, unit_hx, unit_hy, num_plane_spins, num_replicas, field_energies
    )

    
    threads_per_block_m = Int32(256)  # Max threads per block
    blocks_per_replica_m = Int32(cld(num_plane_spins, threads_per_block_m))  # Multiple blocks per replica
    grid_size_m = (Int32(num_replicas), Int32(blocks_per_replica_m))  # 2D grid
    shared_mem_size_m = sizeof(Float32) * threads_per_block_m  # Shared memory per block
    
    # **Launch Kernel**
    @cuda threads=(Int32(1), threads_per_block_m) blocks=grid_size_m shmem=shared_mem_size_m compute_magnetization_kernel!(
        SX, SY, unit_hx, unit_hy, num_plane_spins, num_replicas, magnetizations
    )


    #### Hysteresis Loop Parameters  ####

    hyst_range = vcat(collect(0.0f0:hstep:H_max), collect(H_max:-hstep:-H_max), collect(-H_max:hstep:H_max))
    lh = length(hyst_range)

    #### Preallocate Storage on CPU for Results. Using Float16 for lighter data files.  ####

    collected_magnetizations = zeros(Float16, num_replicas, 1, mcSteps, lh)
    collected_Energy_db = zeros(Float16, num_replicas, 1, mcSteps, lh)
    collected_Energy_h = zeros(Float16, num_replicas, 1, mcSteps, lh)
    collected_Energy = zeros(Float16, num_replicas, 1, mcSteps, lh)
    SX_configs = zeros(Float16, num_spins, num_replicas, lh)
    SY_configs = zeros(Float16, num_spins, num_replicas, lh)

    ####  Preallocate Random Number Buffers  ####

    metro_rand_vals = CUDA.rand(Float32, num_plane_spins, num_replicas)  # (i, r)
    rand_indices = CUDA.zeros(Int32, num_plane_spins, num_replicas)  # (i, r)


    ####  Start Hysteresis Loop  ####

    for (i, h) in enumerate(hyst_range)
        if i%79 == 0
            println("#####################   Running simulation at H = $h   ######################")
        end
	flush(stdout)

        for mc_step = 1:mcSteps
            
            # Generate fresh random values (between 0 and 1) for Metropolis criterion (Reused Buffer from before the loop)
            metro_rand_vals .= CUDA.rand(Float32, num_plane_spins, num_replicas)
            # Generate fresh random spin indices to flip in this MC step (Reused Buffer from before the loop)
            rand_floats = CUDA.rand(Float32, num_plane_spins, num_replicas)  
            rand_indices .= trunc.(Int32, 1 .+ rand_floats .* num_plane_spins) 

        
            # Run Metropolis (Modifies global memory GPU Arrays in Place)
            metropolis_step!(J_LL, J_RR, J_LR, J_RL, S, SX, SY, neighbor_matrix, 
                dumbell_energies, field_energies, magnetizations, rand_indices, metro_rand_vals, 
                h, unit_hx, unit_hy, T, num_replicas, num_spins, num_plane_spins, max_neighbors
            )

            # Store Results in CPU Memory
            # each snapshot is stored in the correct location in the 4D or 3D array   
            collected_magnetizations[:, :, mc_step, i] .= Array{Float16}(magnetizations)
            collected_Energy_db[:, :, mc_step, i] .= Array{Float16}(dumbell_energies)
            collected_Energy_h[:, :, mc_step, i] .= Array{Float16}(field_energies)
            SX_configs[:, :, i] .= Array{Float16}(SX)
            SY_configs[:, :, i] .= Array{Float16}(SY)
        end
    end

    collected_Energy .= collected_Energy_db .- collected_Energy_h

    println("Simulation complete!")

    # Save Results to Disk (Copying to CPU at End)
    control_label = use_controls ? "Controlled" : "NonControlled"

    save_object("M_data_$(num_plane_spins)_spins_$(control_label)_NumRepls_$(num_replicas)_MCsteps_$(mcSteps).jld", Dict("M" => collected_magnetizations))
    save_object("E_data_$(num_plane_spins)_spins_$(control_label)_NumRepls_$(num_replicas)_MCsteps_$(mcSteps).jld", Dict("E" => collected_Energy, "E_db" => collected_Energy_db, "E_h" => collected_Energy_h))
    save_object("Configs_data_$(num_plane_spins)_spins_$(control_label)_NumRepls_$(num_replicas)_MCsteps_$(mcSteps).jld", Dict("SX" => SX_configs, "SY" => SY_configs))


    return
end

# simulate_system_hysteresis(num_plane_spins::Int32, num_replicas::Int32, use_controls::Bool, mcSteps::Int32)
@time simulate_system_hysteresis(Int32(400), Int32(100), false, Int32(1000));

CUDA.device_reset!() 
