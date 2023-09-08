class Responses:
    eigen = """
        The eigenvalues and eigenvectors of A'A and AA' are related, but 
        they represent different aspects of the matrix. A'A is a square 
        symmetric matrix, so its eigenvalues are real and non-negative, and 
        its eigenvectors are orthogonal. AA' may not be square, so it 
        doesn't necessarily have real eigenvalues or orthogonal 
        eigenvectors.
    """

    inverse_matrix = """
        Whether you can invert matrix A depends on its properties. 
        If the matrix is square and has full rank (i.e., its rows and columns 
        are linearly independent), then you can invert it. To invert matrix A, 
        you can use methods like Gaussian elimination or matrix inversion 
        formulae.
    """

    distance_face = """
        This calculation computes the pixel-wise difference between your face 
        and the average face, and then calculates the magnitude of that 
        difference vector. The resulting value represents the Euclidean 
        distance between the two faces.
    """

    umap = """
        Manifold Learning: UMAP is based on the idea of manifold learning, 
        which aims to discover the underlying structure of high-dimensional 
        data. Manifolds are lower-dimensional, non-linear structures embedded 
        within the high-dimensional space. UMAP focuses on finding a 
        low-dimensional representation of the data while preserving the 
        relationships and structures present in the original high-dimensional 
        space.

        Fuzzy Topological Structure: UMAP builds upon the concept of a fuzzy 
        topological structure. It constructs a weighted, k-nearest neighbor 
        (k-NN) graph where each data point is connected to its nearest 
        neighbors. The weights on the edges of this graph represent the 
        similarity between points. UMAP uses a probability-based approach to 
        determine these weights, which allows for the incorporation of 
        uncertainty into the relationships between data points.
        
        Optimization Objective: UMAP defines an optimization objective that 
        balances two key principles: preserving local neighborhood structures 
        and maintaining global connectivity. It aims to minimize a cost 
        function that quantifies the discrepancy between the original 
        high-dimensional distances and the distances in the low-dimensional 
        space, while also considering the distribution of pairwise similarities.
        
        Stochastic Gradient Descent: UMAP employs stochastic gradient descent 
        (SGD) to minimize the optimization objective. The optimization 
        process adjusts the positions of data points in the low-dimensional 
        space iteratively, seeking to find an embedding that captures both 
        local and global structures.
        
        Fuzzy Sets and Low-Dimensional Embedding: UMAP utilizes concepts 
        from fuzzy set theory to create a low-dimensional embedding. 
        This allows it to capture uncertainty and provide a more flexible 
        representation of data points in the low-dimensional space compared to 
        some other techniques.
    """

    lda = """
        Linear Transformations: LDA focuses on finding linear combinations of 
        features (variables) that best separate different classes or categories
         in the data. It does this by finding linear transformation matrices.

        Maximizing Class Separability: The primary goal of LDA is to maximize 
        the separation between the means (centroids) of different classes while
         minimizing the scatter (variance) within each class. This is done by
         defining an objective function that quantifies class separability.
        
        Eigenvalue Decomposition: LDA involves eigenvalue decomposition of 
        matrices. Specifically, it calculates the eigenvalues and eigenvectors
         of the scatter matrices, such as the within-class scatter matrix and 
         between-class scatter matrix.
        
        Fisher's Criterion: LDA optimizes Fisher's criterion, which is defined 
        as the ratio of the between-class variance to the within-class variance. 
        Maximizing this ratio results in better class separation.
        
        Dimension Reduction: After calculating the eigenvalues and eigenvectors, 
        LDA selects a subset of the eigenvectors (discriminants) corresponding 
        to the largest eigenvalues. These discriminants form a new feature space 
        with reduced dimensionality.
    """
