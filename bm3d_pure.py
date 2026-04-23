"""
Implementation of BM3D image denoising algorithm.
Pure, intro-level hand coded python only for educational purposes
Reference: Dabov et al., "Image Denoising by Sparse 3D Transform-Domain
           Collaborative Filtering", TIP 2007.
Author: RV

time spent: 03/31/2026 - 04/22/2026
"""
import math
from PIL import Image
from AWGN import AWGN

# Stage 1 parameters
BLOCK_SIZE_1 = 8 # block side length
MAX_GROUP_SIZE_1 = 16 # max blocks per group
TAU_MATCH_1 = 2500 # max allowed dissimilarity
STEP_1 = 3 # sliding step for reference blocks

# Stage 2 Global Variables
BLOCK_SIZE_2 = 8 # block side length
MAX_GROUP_SIZE_2 = 32 # max blocks per group
TAU_MATCH_2 = 400 # max allowed dissimilarity
STEP_2 = 3 # sliding step for reference blocks

# Global Variables used in both stages
SEARCH_WIN = 39 # search window side length
LAMBDA_HT = 2.7 # hard threshold multiplier
LAMBDA_DIST = 2.5 # pre-filter threshold multiplier (distance)

SIGMA = 25

# 1-D Transforms

def dct1d(x):
    """Discrete Cosine Transform 1 dimenstional.
    Applies row-wise then column-wise
    X[k] = w(k) * sigma_{n=0}^{N-1} x[n] * cos(pi(2n+1)k / 2N)
        where  w(0) = sqrt(1/N),  w(k>0) = sqrt(2/N)

    Parameters:
        x - list of floats, length N.

    Returns:
        list of floats, same length.
    """
    N = len(x)
    X = []
    for k in range(N):
        s = 0
        for n in range(N):
            s += x[n] * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
        if k == 0:
            w = math.sqrt(1.0 / N)
        else:
            w = math.sqrt(2.0 / N)
        X.append(w * s)
    return X

def idct1d(X):
    """Inverse of dct1d. Reconstructs pixel values from DCT coefficients after thresholding.
    x[n] = simga_{k=0}^{N-1} w(k) * X[k] * cos(pi(2n+1)k / 2N)

    Parameters:
        X - list of floats, length N.

    Returns:
        list of floats
    """
    N = len(X)
    x = []
    for n in range(N):
        s = math.sqrt(1.0 / N) * X[0]
        for k in range(1 , N):
            s += math.sqrt(2.0 / N) * X[k] * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
        x.append(s)
    return x

def wht1d(x):
    """Walsh Hadamard Transform. Length of x must be a power of 2.
    Applied across the 3D stack of matched blocks

    Parameters:
        x - list of floats.

    Returns:
        list of floats.
    """
    N = len(x)
    v = list(x)
    h = 1
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                a = v[j]
                b = v[j + h]
                v[j] = a + b
                v[j + h] = a - b
        h = h * 2
    s = 1.0 / math.sqrt(N)
    result = []                                                                                                           
    for val in v:                                                                                                       
        result.append(val * s)                                                                                            
    return result

def iwht1d(X):
    """Inverse of wht1d. (IDENTICAL!!!)
    Parameters:
        X - list of floats.
    Returns:
        list of floats.
    """
    return wht1d(X)

# 2-D transforms — delegated to scipy for speed (O(N log N) vs O(N²))

def dct2d(block):
    """2D DCT on a block. uses separability to apply dct1d to each row, then to each column.

    Parameters:
        block - list of lists of floats size N x N.

    Returns:
        list of lists of floats
    """
    N, M = len(block), len(block[0])
    temp = []
    for i in range(N):
        temp.append(dct1d(block[i]))

    result = []
    for i in range(N):
        result.append([0.0] * M)

    for j in range(M):                                                                                                    
      col_vals = []                                 
      for i in range(N):
          col_vals.append(temp[i][j])  # collect the j-th value from each row
      col = dct1d(col_vals)                                                    
      for i in range(N):
          result[i][j] = col[i]                        
    return result 

def idct2d(block):
    """2D inverse DCT on a block.
    uses separability to applies idct1d to each column, then to each row.

    Parameters:
        block - list of lists of floats, shape (N, N).

    Returns:
        list of lists of floats, same shape.
    """
    N, M = len(block), len(block[0])                                                                                                 
    temp = []
    for i in range(N):
        temp.append([0.0] * M)
    for j in range(M):                                                                                                    
        col_vals = []                                                                                                     
        for i in range(N):
            col_vals.append(block[i][j])                                                            
        col = idct1d(col_vals)
        for i in range(N):                                                                                                
            temp[i][j] = col[i]
    result = []
    for i in range(N):
        result.append(idct1d(temp[i]))
    return result

#helpers

def extract_block(im, row, col, block_size):
    """Extract a square block from the image at a given position.

    Parameters:
        image - 2D list of floats (grayscale image).
        row - top-left row index of the block.
        col - top-left column index of the block.
        block_size - side length of the square block.

    Returns:
        list of lists of floats, block_size, block_size size
    """
    result = [] #8x8 image
    for i in range(block_size):
        result.append([0.0]*block_size)

    for v in range(row, row + block_size,1):
        for u in range(col, col + block_size,1):
            result[v - row][u - col] = im[v][u]

    return result

def block_dissimilarity(ref_t, cand_t, sigma, lambda_dist, block_size):
    """calculate the normalised dissimilarity between two pre-transformed blocks. 
    Zero out coefficients below the pre-filter threshold before
    comparing (hard thresholding at lambda_dist * sigma).

        d(Z_xR, Z_x) = ||T'_2D(Z_xR) - T'_2D(Z_x)||^2 / N1^2

        this uses Parseval's Theorem: The total energy of signal calculated
        by summing its euclidian quared amplitudes in the time domain equals
        the total engery calulated in the frequency domain.

    example of low dissimilarity:
        ref_block has all pixel values at 100.0 and cand_block has pixel
        values all at 102.0, their dissimilarity would be 4.0
    example of high dissimilarity:
        ref_block has all pixel values at 0.0 and cand_block has pixel
        values all at 200.0, their dissimilarity would be 40,000

    Parameters:
        ref_t - pre-computed dct2d of the reference patch Z_xR.
        cand_t - pre-computed dct2d of the candidate patch Z_x.
        sigma - standard deviation float
        lambda_dist - threshold multiplier for the pre-filtering step.
        block_size - length and width amount as integer

    Returns:
        Scalar dissimilarity value.
    """

    ht = lambda_dist * sigma #hard thresholing: chicken in the egg paradox

    total = 0.0
    for i in range(block_size):
        for j in range(block_size):
            if abs(ref_t[i][j]) > ht:
                ref = ref_t[i][j]
            else:
                ref = 0.0
            if abs(cand_t[i][j]) > ht:
                cand = cand_t[i][j]
            else:
                cand = 0.0
            total += (ref - cand) ** 2
    n_dissimilarity = total / (block_size**2)

    return n_dissimilarity

# Step One: Grouping

def group_match(im, ref_row, ref_col, block_size, search_win, max_group_size, tau_match, sigma, lambda_dist):
    """Finds similar blocks and stacks them into a 3D group.

    Searches a search_win x search_win buffer centred on
    ref_row and ref_col and ranks candidates by block_dissimilarity, and keeps up
    to max_group_size blocks whose dissimilarity is ≤ tau_match:

        S_xR = { x in X : d(Z_xR, Z_x) <= tau_match}

    parameters:
        image - 2D list of floats.
        ref_row - top-left row of the reference block.
        ref_col - top-left column of the reference block.
        block_size - side length of each block.
        search_win - side length of the search neighbourhood.
        max_group_size - maximum number of blocks in the returned group.
        tau_match - maximum allowed dissimilarity threshold.
        sigma - noise standard deviation.
        lambda_dist - pre-filtering threshold multiplier for distance.

    Returns touple:
        group - list of blocks, each block a list of lists of floats,
            shape num_matches, block_size, block_size.
        positions - list of row, col tuples for each block in the group,
            in the same order as group's first axis.
    """
    N = len(im)
    M = len(im[0])
    search_win_r = search_win // 2
    # determine the search window bounds clamped to image edges:
    row_start = max(0, ref_row - search_win_r)
    # if (ref_row - search_win_r), use 0
    row_end = min(ref_row + search_win_r, N - block_size + 1)
    # if ref_row + search_win_r > N - block_size + 1, use N - block_size + 1
    col_start = max(0, ref_col - search_win_r)
    col_end = min(ref_col + search_win_r, M - block_size + 1)

    ref_block = extract_block(im, ref_row, ref_col, block_size)
    ref_t = dct2d(ref_block) # pre-compute once; reused for every candidate in this window

    # candidates stores (disim, u, v, cand_t) so DCTs are cached alongside positions
    candidates = []
    for v in range(col_start, col_end, 1):
        for u in range(row_start, row_end, 1):
            if u == ref_row and v == ref_col:
                candidates.append((0.0, u, v, ref_t)) # reference always wins
                continue
            cand_block = extract_block(im, u, v, block_size)
            cand_t = dct2d(cand_block)
            disim = block_dissimilarity(ref_t, cand_t, sigma, lambda_dist, block_size)
            if disim <= tau_match:
                candidates.append((disim, u, v, cand_t))

    #Keep only candidates with dissimilarity <= tau_match, sorted best-first
    candidates.sort(key=lambda x: x[0])
    ''''
    lambda x: x[0] is an anonymous function that takes one argument x, a single tuple,
    and returns x[0] the first element disim. For each tuple in the list, 
    Python calls lambda to get a sorting key, then ranks tuples by that key ascending
    ex.
    lambda (4.2, 3, 5) returns 4.2
    lambda (0.0, 2, 2) returns 0.0
    lambda (1.1, 4, 4) returns 1.1
    sorted as:
    [(0.0, 2, 2), (1.1, 4, 4), (4.2, 3, 5)]
    '''
    group = []
    positions = []
    length = 0
    if max_group_size < len(candidates):
        length = max_group_size
    else:
        length = len(candidates)

    for i in range(length):                                                                                             
        u = candidates[i][1]                     
        v = candidates[i][2]
        group.append(extract_block(im, u, v, block_size))                                                                 
        positions.append((u, v))

    return group, positions

# step Two: 3-D Transform / Inverse

def transform_3d(group):
    """Apply a separable 3D transform to a group of stacked blocks.
    Step 1: apply a 2D DCT to each block
    Step 2: apply a 1D Walsh-Hadamard Transform (wht1d) along the grouping
            axis (across corresponding coefficients of all blocks).

    parameters:
        group - list of blocks, shape (num_blocks, block_size, block_size).

    Returns:
        list of blocks of transform coefficients, same shape as input.
    """
    num_blocks, block_sl = len(group), len(group[0])
    # pad to next power of 2
    p = 1
    while p < num_blocks:
        p  = p * 2

    # For each coefficient position (i, j), collect that same position
    # from every block into a 1D list, apply wht1d to that list.
    result = []
    for i in range(num_blocks):
        temp = []
        for j in range(block_sl):
            temp.append([0.0] * block_sl)
        result.append(temp)

    dct_blocks = []                                                                                                       
    for block in range(num_blocks):                                                                                       
        dct_blocks.append(dct2d(group[block]))            
                                                                                                                            
    for i in range(block_sl):
        for j in range(block_sl):                                                                                         
            column = []                                   
            for block in range(num_blocks):
                column.append(dct_blocks[block][i][j])
            column += [0.0] * (p - num_blocks)                                                                            
            transformed = wht1d(column)
            for block in range(num_blocks):                                                                               
                result[block][i][j] = transformed[block]

    return result

def i_transform_3d(coeffs):
    """Apply the inverse separable 3D transform to recover spatial-domain blocks.

    Inverse of transform_3d: first inverse WHT across blocks, then inverse
    2D DCT on each block.

    parameters:
        coeffs - list of blocks of transform coefficients,
                shape num_blocks, block_size, block_size.

    Returns:
        list of blocks in the spatial domain, same shape.
    """
    num_blocks, block_sl = len(coeffs), len(coeffs[0])

    # pad to next power of 2 (must match what transform_3d did)
    p = 1
    while p < num_blocks:
        p = p * 2

    # inverse WHT across blocks at each (i, j) position
    result = []
    for block in range(num_blocks):
        temp = []
        for i in range(block_sl):
            temp.append([0.0] * block_sl)
        result.append(temp)

    for i in range(block_sl):
        for j in range(block_sl):
            column = []
            for block in range(num_blocks):
                column.append(coeffs[block][i][j])
            column += [0.0] * (p - num_blocks)
            inverse = iwht1d(column)
            for block in range(num_blocks):
                result[block][i][j] = inverse[block]
    final =[]
    # inverse 2D DCT each block
    for block in result:
        final.append(idct2d(block))

    return final

# Step Three: Filtering

def hard_threshold(coeffs, threshold):
    """Apply hard thresholding to 3D transform coefficients (Stage 1 filtering).
    Sets to zero every coefficient whose absolute value is below `threshold`:

    if |x| >= threshold, T(x) = x else  0

    parameters:
        coeffs - list of blocks of transform coefficients.
        threshold - scalar threshold (typically lambda_ht * sigma).

    returns touple:
        thresholded - list of blocks with small coefficients zeroed out.
        n_nonzero - number of surviving (non-zero) coefficients; used to
                    compute the aggregation weight w^ht_xR = 1 / n_nonzero.
    """
    num_blocks, block_sl = len(coeffs), len(coeffs[0])


    thresholded = []
    for block in range(num_blocks):
        temp = []
        for i in range(block_sl):
            temp.append([0.0] * block_sl)
        thresholded.append(temp)
    n_nonzero = 0

    for block in range(num_blocks):
        for v in range(block_sl):
            for u in range(block_sl):
                if abs(coeffs[block][u][v]) >= threshold:
                    thresholded[block][u][v] = coeffs[block][u][v]
                    n_nonzero += 1
                #else:
                #   already 0.0

    if n_nonzero == 0:
        n_nonzero = 1
        # avoids ZeroDivisionError if all coefficients are below threshold

    return thresholded, n_nonzero

def wiener_filter(coeffs_noisy, coeffs_basic, sigma):
    """Apply Wiener filtering to noisy 3D transform coefficients (Stage 2).
     Uses the basic estimate to derive coefficient shrinkage weights:

        sigma_u_sq = max(|Y_basic_k|^2 - sigma^2, 0)
        W_k = sigma_u_k^2 / (sigma_u_k^2 + sigma^2)

    Returns the filtered coefficients W_k * Y_noisy_k and the squared
    L2 norm of the weights (used for the aggregation weight w^wie_xR).

    parameters:
        coeffs_noisy - transform coefficients of the noisy group.
        coeffs_basic - ltransform coefficients of the basic estimate group.
        sigma - noise standard deviation.

    Returns touple:
        filtered - list of blocks of Wiener-filtered coefficients.
        weights - scalar W.W used for the aggregation weight
                    w^wie_xR = weights_l2 / sigma^2.
    """
    num_blocks, block_sl = len(coeffs_noisy), len(coeffs_noisy[0])

    filtered = []
    for i in range(num_blocks):
        temp = []
        for j in range(block_sl):
            temp.append([0.0] * block_sl)
        filtered.append(temp)

    weights_l2 = 0.0

    for block in range(num_blocks):
        for v in range(block_sl):
            for u in range(block_sl):
                ''' sigma_u_sq = (Y_basic_k)**2 - sigma**2)
                This estimates the true signal power at coefficient k.
                The basic estimate contains signal + extra noise, so
                subtract out the noise variance sigma**2.'''
                b = coeffs_basic[block][u][v] # Y_basic_k, the basic coefficient
                sigma_u_sq = b**2 - sigma**2 # *(Y_basic_k)**2 - sigma**2
                # how much of this coefficient is signal versus noise?
                if sigma_u_sq < 0:
                    sigma_u_sq = 0 #ensure Y_basic_k positive
                # compute shrinkage weight
                if sigma_u_sq == 0:
                    W = 0
                else:
                    W = sigma_u_sq / (sigma_u_sq + sigma**2) # W_k = sigma_u_k**2 / (sigma_u_k**2 + sigma**2)
                    '''When signal power sigma_u_sq is large relative to noise sigma**2,
                    W = 1 (keep the coefficient). When signal power is zero, W = 0 (kill the coefficient).'''
                # store the filtered coefficient
                filtered[block][u][v] = W * coeffs_noisy[block][u][v]
                '''coeff_basic only provided the weight. The actual
                data being filtered is the coeff_noisy.'''
                weights_l2 += W**2 # accumulates ||W||**2 (weight norm) across all coefficients in the group

    return filtered, weights_l2

# step Four: Aggregation

def aggregate(numerator, denominator, filtered_group, positions, weight, block_size):
    """Accumulate overlapping filtered blocks directly into numerator/denominator buffers.

    Aggregation formula:
        y_hat(x) = sigma_{xR} sigma__{xm ing S_xR}  w_xR * Y_hat^xR_xm(x) /
               sigma__{xR} sigma__{xm in S_xR}  w_xR * 1_{x in patch(xm)}

    parameters:
        numerator - 2D list of floats - weighted-sum buffer.
        denominator - 2D list of floats -  weight buffer
        filtered_group - list of spatial-domain blocks after inverse transform.
        positions - list of (u, v) tuples, one per block.
        weight - scalar weight for this group (w^ht_xR or w^wie_xR).
        block_size - side length of each block.
    returns:
        none
    """
    block_idx = 0                                
    for pos in positions:
        row = pos[0]                                                                                                      
        col = pos[1]
        for v in range(block_size):                                                                                       
            for u in range(block_size):
                numerator[row + v][col + u]   += weight * filtered_group[block_idx][v][u]
                denominator[row + v][col + u] += weight                                                                   
        block_idx += 1

# Stage Orchestrators

def bm3d_stage1(noisy, sigma):
    """Stage 1: basic estimate via collaborative hard-thresholding.

    Iterates over all reference block positions (stride = STEP_1):
        1 - group_match - find and stack similar blocks from `noisy`.
        2 - transform_3d - 3D transform the group (reuses cached DCTs from group_match).
        3 - hard_threshold - threshold at LAMBDA_HT * sigma.
        4 - inverse_transform_3d - back to spatial domain.
        5 - aggregate - accumulate weighted blocks directly into shared buffers.

    Divides the numerator buffer by the denominator buffer to get the
    basic estimate.

    parameters:
        noisy - 2D list of floats, noisy grayscale image, values in [0, 255].
        sigma - estimated noise standard deviation.

    Returns:
        basic_estimate - 2D list of floats, same shape as noisy.
    """

    N, M = len(noisy), len(noisy[0])

    numerator = []
    for i in range(N):
        numerator.append([0.0] * M )
    denominator = []
    for i in range(N):
        denominator.append([0.0] * M)
    basic_estimate = []
    for i in range(N):
        basic_estimate.append([0.0] * M)

    for ref_row in range(0, N - BLOCK_SIZE_1 + 1, STEP_1):
        for ref_col in range(0, M - BLOCK_SIZE_1 + 1, STEP_1):

            group, positions = group_match(noisy, ref_row, ref_col,
                            BLOCK_SIZE_1, SEARCH_WIN, MAX_GROUP_SIZE_1,
                            TAU_MATCH_1, sigma, LAMBDA_DIST)
            '''Returns:
                group:list of blocks, each block a list of lists of floats,
                    shape num_matches, block_size, block_size.
                 positions: list of row, col tuples for each block in the group,
                            in the same order as group's first axis.'''
            coeffs = transform_3d(group)
            '''Returns list of blocks of transform coefficients, same shape as input.'''
            ht = hard_threshold(coeffs, LAMBDA_HT * sigma)
            '''Returns:
                thresholded: list of blocks with small coefficients zeroed out.
                n_nonzero: number of surviving (non-zero) coefficients; used to
                    compute the aggregation weight w^ht_xR = 1 / n_nonzero.'''
            filtered_group = i_transform_3d(ht[0])
            '''Returns:
                list of blocks in the spatial domain, same shape.'''
            aggregate(numerator, denominator, filtered_group, positions, 1.0/ht[1], BLOCK_SIZE_1)

    for v in range(N):
        for u in range(M):
            if denominator[v][u] != 0.0:
                basic_estimate[v][u] = numerator[v][u] / denominator[v][u]
            else:
                basic_estimate[v][u] = noisy[v][u]

    return basic_estimate

def bm3d_stage2(noisy, basic_estimate, sigma):
    """Stage 2: final estimate via empirical Wiener filtering.

    Uses basic_estimate as a basic signal. Iterates over all reference
    block positions (stride = STEP_2):
        1 - group_match - find similar blocks using the basic estimate.
        2 - transform_3d - 3D transform both the noisy group and the basic group
        3 - wiener_filter - filter noisy coefficients with pilot-derived weights.
        4 - inverse_transform_3d - back to spatial domain.
        5 - aggregate - accumulate weighted blocks directly into shared buffers.
    Divides numerator by denominator to get the final estimate.

    parameters:
        noisy - 2D list of floats, original noisy image.
        basic_estimate - 2D list of floats from Stage 1.
        sigma - estimated noise standard deviation.

    Returns:
        final_estimate - 2D list of floats, same shape as noisy.
    """

    N, M = len(noisy), len(noisy[0])

    
    numerator = []
    for i in range(N):
        numerator.append([0.0] * M )
    denominator = []
    for i in range(N):
        denominator.append([0.0] * M)
    final_estimate = []
    for i in range(N):
        final_estimate.append([0.0] * M)

    for ref_row in range(0, N - BLOCK_SIZE_2 + 1, STEP_2):
        for ref_col in range(0, M - BLOCK_SIZE_2 + 1, STEP_2):

            group_basic, positions = group_match(basic_estimate, ref_row, ref_col,
                              BLOCK_SIZE_2, SEARCH_WIN, MAX_GROUP_SIZE_2,                                               
                              TAU_MATCH_2, sigma, LAMBDA_DIST)

            group_noisy = []                                                                                                      
            for pos in positions:                                 
                u = pos[0]
                v = pos[1]
                group_noisy.append(extract_block(noisy, u, v, BLOCK_SIZE_2))

            coeffs_basic = transform_3d(group_basic)
            # group_dcts cached from group_match; no redundant dct2d calls for the basic group
            coeffs_noisy = transform_3d(group_noisy)

            filtered, weights_l2 = wiener_filter(coeffs_noisy, coeffs_basic, sigma)

            filtered_group = i_transform_3d(filtered)

            aggregate(numerator, denominator, filtered_group, positions, weights_l2/sigma**2, BLOCK_SIZE_2)

    for v in range(N):
        for u in range(M):
            if denominator[v][u] != 0.0:
                final_estimate[v][u] = numerator[v][u] / denominator[v][u]
            else:
                final_estimate[v][u] = basic_estimate[v][u]

    return final_estimate


# final product

def bm3d(noisy_image, sigma):
    """Run the full two-stage BM3D denoising pipeline.

    Stage 1 (hard-thresholding) produces a basic estimate.
    Stage 2 (Wiener filtering)  uses that estimate as a pilot to produce
    the final, higher-quality denoised result.

    parameters:
        noisy_image - 2D list of floats (grayscale), values in [0, 255].
        sigma - estimated noise standard deviation.

    Returns:
        denoised - 2D list of floats, same shape as noisy_image.
    """
    basic_estimate = bm3d_stage1(noisy_image, sigma)
    # returns basic_estimate, a 2D list of floats, same shape as noisy.
    denoised = bm3d_stage2(noisy_image, basic_estimate, sigma)
    return denoised

# Main Function

def main():

    # Load image
    im = Image.open("mandrill.jpg").convert("L")
    M, N = im.size
    name = "mandrill"

    # Pillow -> 2D float list
    image = []
    for v in range(N):
        row =[]
        for u in range(M):
            row.append(float(im.getpixel((u, v))))
        image.append(row)

    # Add noise
    noisy = AWGN(SIGMA).apply(image)
    noisy_img = Image.new('L', (M, N))
    for v in range(N):
        for u in range(M):
            noisy_img.putpixel((u, v), int(max(0, min(255, noisy[v][u]))))
    noisy_img.save(f"noisy_{name}_{SIGMA}.jpg")

    # Denoise
    denoised = bm3d(noisy, SIGMA)
    denoised_img = Image.new('L', (M, N))
    for v in range(N):
        for u in range(M):
            denoised_img.putpixel((u, v), int(max(0, min(255, denoised[v][u]))))
    denoised_img.save(f"denoised_{name}_{SIGMA}.jpg")

    return

if __name__ == "__main__":
    main()