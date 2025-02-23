# Image Processing Transformations and Distance Functions

## Training and Runtime Transformations:
1. **Centering (Mean Removal)**
2. **Principal Component Analysis (PCA)**
3. **Deskew (Tilt Correction)** - Fix existing code
4. **Fourier Transform Features** - worst with KNN (test ac = 93.28)

## Training-Only Transformations:
1. **Elastic Deformations?**
2. **Morphological Operations** (Potentially complex)
3. **Zernike Moments**

## Distance Functions:
1. **Image Distortion Model (IDM)**
2. **Earth Mover's Distance (EMD)**
3. **L2 (Euclidean Distance)**
4. **L1 (Manhattan Distance)**
5. **Tangent Distance**
6. **Cosine Distance** - nice, with KNN (test ac 97.33)