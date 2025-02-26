# Image Processing Transformations and Distance Functions

## Training and Runtime Transformations:
1. **Centering (Mean Removal)** - it seems the images are centered
2. **Principal Component Analysis (PCA)** - KNN: variance=.85, test_acc=97.52, BETTER
3. **Deskew (Tilt Correction)** - Fix existing code
4. **Fourier Transform Features** - worse with KNN (test ac = 93.28)

## Training-Only Transformations:
1. **Elastic Deformations?**
2. **Morphological Operations** (Potentially complex)
3. **Zernike Moments** - very bad ( less than 0.6 on svm,knn,random-forest)

## Distance Functions:
1. **Image Distortion Model (IDM)** - too heavy
2. **Earth Mover's Distance (EMD)** - too heavy
3. **L2 (Euclidean Distance)** - I think that's the default of knn (so no dif from the basic model)
4. **L1 (Manhattan Distance)** -  worse (test ac 96.33) 
5. **Tangent Distance** - too heavy
6. **Cosine Distance** - A little better (test ac 97.33)

need to add:
- description for every transformation - done
- description for every distance function - done
- summery of accuracy
- a description and conclusion file