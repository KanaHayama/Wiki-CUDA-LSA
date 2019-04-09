# Platfrom

+ x64
+ Windows 10
+ Visual Studio 2017
+ CUDA 10.0

# Sub-project Organization

1. WikiExtractAndFormatting: Extract articles from Wiki article XML file. Output formatted document-text mapping.
2. StemmingAndLemmatization: Stemming and Lemmatization for each article. Output formatted document-text mapping.
3. ParallelTDIDF: Generate TD matrix then compute IDF matrix. Output a sparse matrix (sparse or dense? depending on experimental results).
4. LargeMatrixParallelSVD: Implement parallel SVD for large matrix. Output three matrices.
5. LatentSemanticAnalysis: Use SVD to compute term-term correlation matrix.