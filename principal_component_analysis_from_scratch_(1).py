## Part 1 : Principal Component Analysis scratch implementation

##### 1. Resize Images
- Standardize all images to **64 × 64 × 3** dimensions.

##### 2. Implement PCA (Eigen vs. SVD)
- Create a `PCA` class that includes:
  - **Eigen decomposition** method.
  - **Singular Value Decomposition (SVD)** method.

##### 3. Channel-wise PCA
- Split images into **Red, Green, and Blue (R, G, B)** channels.
- Apply PCA separately to each channel.

##### 4. Grayscale PCA
- Convert images to **grayscale**.
- Apply PCA to the grayscale images.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
import os

import os
print(os.listdir("/kaggle/input/"))

class PCA:
    def __init__(self, n_components, method = "svd"):
        """
        PCA initialization function
        Parameters :
        - n_components : no. of principal components to keep (same as sklearn variable)
        """

        self.n_components = n_components
        self.method = method
        self.mean = None
        self.topcomponents = None
        self.X_centered = None
        self.explained_variance_ratio_ = None

    def fit_eigen(self, X):
        """Eigen value decomposition algorithm"""
        cov = np.cov(X.T)  #computing covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov) #finding eigen values and corresponding eigen vectors
        eigen_index_pair = [(idx, val) for idx, val in enumerate(eigenvalues)] #pairing index value with each eigen value

        #sorting eigen values and eigen vectors in descending order
        #sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigen_pair = sorted(eigen_index_pair, key = lambda x: x[1], reverse = True)
        sorted_indices = [idx for idx, val in sorted_eigen_pair]
        eigenvectors = eigenvectors[:, sorted_indices]

        #Selecting top components as per input n_components
        self.topcomponents = eigenvectors[:, :self.n_components]
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_variance

    def fit_svd(self, X):
        """Singular value decomposition algorithm"""

        U, S, VT = np.linalg.svd(X, full_matrices = False)

        #selecting top components
        self.topcomponents = VT[:self.n_components]
        total_variance = np.sum(S**2)
        self.explained_variance_ratio_ = (S[:self.n_components]**2) / total_variance

    def fit(self, X):
        #Centering the data
        self.mean = np.mean(X, axis=0) #Calculate mean intensity of pixels of each image
        self.X_centered = X - self.mean

        if self.method == "eigen":
            self.fit_eigen(self.X_centered)
        elif self.method == "svd":
            self.fit_svd(self.X_centered)
        else:
            raise ValueError("Invalid method")

    def transform(self,X):
        """ Projecting dataset X onto Principal components """
        if self.method == "eigen":
            transformed_data = np.dot(self.X_centered, self.topcomponents)
        elif self.method == "svd":
            transformed_data = np.dot(self.X_centered, self.topcomponents.T)
        return transformed_data.real

    def inverse_transform(self, transformed_data):
        if self.method == "eigen":
             reconstructed_data = np.dot(transformed_data.real, self.topcomponents.T) + self.mean
        elif self.method == "svd":
             reconstructed_data = np.dot(transformed_data.real, self.topcomponents) + self.mean
        return reconstructed_data

"""### Image resizing and processing

Information about the images :
The original shape of the images are 256 x 256 x 3. We will resize them to 64 x 64 along with applying anti-aliasing
"""

image_path = "/kaggle/input/mlds-assignmet-2-ml-dl/Dataset/train/images"
images = []
for filename in os.listdir(image_path):
    file_path = os.path.join(image_path, filename)
    if os.path.isfile(file_path) and file_path.lower().endswith('.png'):
        image = io.imread(file_path)
        image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
        images.append(image_resized) #Storing the images as 1D array to images
images = np.array(images) #converting the images list into numpy array

print (image.shape, image_resized.shape, images.shape)

def plot_image_grid(images, title, num_images=10):
    """
    Plots a grid of the first few images in the dataset.

    Parameters:
    - images: NumPy array of shape (num_images, height, width)
    - title: Title for the plot
    - num_images: Number of images to display (default=10)
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # 2 rows, 5 columns

    fig.suptitle(title, fontsize=16)  # Set main title

    for i, ax in enumerate(axes.flat):  # Flatten the 2D axes array
        if i < len(images):
            ax.imshow(images[i])
            ax.axis("off")  # Hide axes

    plt.show()

plot_image_grid(images, "Resized images")

"""### Splitting the RGB channels into individual channels and separately into grayscale"""

images_red = images [:, :, :, 0]
images_green = images [:, :, :, 1]
images_blue = images [:, :, :, 2]
print (images_blue.shape, images_green.shape, images_red.shape)

images_gray = color.rgb2gray(images)
print(images_gray.shape)

plot_image_grid(images_blue, "Blue Channel Images")
plot_image_grid(images_red, "Red Channel Images")
plot_image_grid(images_green, "Green Channel Images")

"""### Applying PCA to different channels

#### 1. Red channel
"""

def run_PCA (X, method, n_components):
    a,b,c = X.shape
    flattened_images = X.reshape(a, b*c) #Flattening the images from into one dimension (909 x 64 x 64 -> 909 x 4096)
    pca = PCA(n_components, method = method)
    pca.fit(flattened_images)
    var_ratio = pca.explained_variance_ratio_
    transform = pca.transform(flattened_images)
    reconstructed = pca.inverse_transform(transform.real)
    principal_components = pca.topcomponents
    return var_ratio, reconstructed, principal_components

variance_ratio, pca_red, _ = run_PCA(images_red, "eigen", 100)
pca_red = pca_red.reshape(-1, 64, 64)
plot_image_grid(pca_red.real, "Red channel images after PCA (eigen)")
print("Total explained variance ratio (eigen) : ", np.sum(variance_ratio))

variance_ratio, pca_red_svd, _ = run_PCA(images_red, "svd", 100)
pca_red_svd = pca_red_svd.reshape(-1, 64,64)
plot_image_grid(pca_red_svd, "Red channel images after PCA (svd)")
print("Total explained variance ratio (svd) : ", np.sum(variance_ratio))

"""#### SVD vs Eigen Value decomposition
For the red channel dataset, eigen value decomposition took longer time tha SVD, given the dataset size (909 x 4096). This is because SVD does not calculate covariance matrix, making it more computationally faster and precise. Therefore, in the further code, we will use SVD implementation.

##
"""

#Finding the appropriate number of components retaining >90% mean variance ratio

fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # Create a row of 4 subplots

# Red channel
var_ratio, _ , _= run_PCA(images_red, "svd", 4096)
percentage_var_explained = var_ratio / np.sum(var_ratio)
cum_var_explained = np.cumsum(percentage_var_explained)
axes[0].plot(cum_var_explained)
axes[0].set_xlabel('n_components (Red)')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('Red Channel PCA')

# Blue channel
var_ratio, _,_ = run_PCA(images_blue, "svd", 4096)
percentage_var_explained = var_ratio / np.sum(var_ratio)
cum_var_explained = np.cumsum(percentage_var_explained)
axes[1].plot(cum_var_explained)
axes[1].set_xlabel('n_components (Blue)')
axes[1].set_title('Blue Channel PCA')

#Green channel
var_ratio, _,_ = run_PCA(images_green, "svd", 4096)
percentage_var_explained = var_ratio / np.sum(var_ratio)
cum_var_explained = np.cumsum(percentage_var_explained)
axes[2].plot(cum_var_explained)
axes[2].set_xlabel('n_components (Green)')
axes[2].set_title('Green Channel PCA')

#Gray channel
var_ratio, _ ,_= run_PCA(images_gray, "svd", 4096)
percentage_var_explained = var_ratio / np.sum(var_ratio)
cum_var_explained = np.cumsum(percentage_var_explained)
axes[3].plot(cum_var_explained)
axes[3].set_xlabel('n_components (Gray)')
axes[3].set_title('Gray Scale PCA')

plt.tight_layout()
plt.show()

"""#### Finding the appropriate number of components
From the plots, we infer that 200 components are sufficient to explain ~95% variance in the dataset, so for the subsequent PCA implementations for all data, we will use 225 principal components.

So, the dataset should reduce from **909 x 64 x 64 x 3** to **909 x 15 x 15 x 3** (after merging rgb channels which is implemented next).
"""

variance_ratio_red, pca_red, PC_red = run_PCA(images_red, "svd", 225)
pca_red = pca_red.reshape(-1, 64, 64)
plot_image_grid(pca_red, "Red channel images after PCA (225 components)")
print("Total explained variance ratio (svd) : ", np.sum(variance_ratio_red))

variance_ratio_blue, pca_blue, PC_blue = run_PCA(images_blue, "svd", 225)
pca_blue = pca_blue.reshape(-1, 64,64)
plot_image_grid(pca_blue, "Blue channel images after PCA (225 components)")
print("Total explained variance ratio (svd) : ", np.sum(variance_ratio_blue))

variance_ratio_green, pca_green, PC_green = run_PCA(images_green, "svd", 225)
pca_green = pca_green.reshape(-1, 64,64)
plot_image_grid(pca_green, "Green channel images after PCA (225 components)")
print("Total explained variance ratio (svd) : ", np.sum(variance_ratio_green))

variance_ratio_grayscale, pca_grayscale, PC_grayscale = run_PCA(images_gray, "svd", 225)
pca_grayscale = pca_grayscale.reshape(-1, 64,64)
plot_image_grid(pca_grayscale, "Grayscale images after PCA (225 components)")
print("Total explained variance ratio (svd) : ", np.sum(variance_ratio_grayscale))

"""#### Combining Red, blue and green channels"""

images_merged = np.stack((pca_red, pca_green, pca_blue), axis=-1)
print(images_merged.shape)
plot_image_grid(images_merged, "Merged images after PCA")

"""#### Transforming test data"""

image_path = "/kaggle/input/mlds-assignmet-2-ml-dl/Dataset/test/images"
images_test = []
for filename in os.listdir(image_path):
    file_path = os.path.join(image_path, filename)
    if os.path.isfile(file_path) and file_path.lower().endswith('.png'):
        image = io.imread(file_path)
        image_resized = resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)
        images_test.append(image_resized)
images_test = np.array(images_test)

print(images_test.shape)
plot_image_grid(images_test, "Test images")

PC_red.shape

n, h, w, c = images_test.shape
images_test_red = images_test[:, :, :, 0].reshape(n, h*w)
images_test_green = images_test [:, :, :, 1].reshape(n, h*w)
images_test_blue = images_test [:, :, :, 2].reshape(n, h*w)
images_test_gray = color.rgb2gray(images_test).reshape(n, h*w)
print (images_test_blue.shape, images_test_green.shape, images_test_red.shape, images_test_gray.shape)

# Transforming the test dataset onto top component matrix found earlier

transform_test_red = images_test_red@PC_red.T
transform_test_blue = images_test_blue@PC_blue.T
transform_test_green = images_test_green@PC_green.T
transform_test_gray = images_test_gray@PC_grayscale.T

print (transform_test_red.shape, transform_test_blue.shape, transform_test_green.shape, transform_test_gray.shape)

#Reconstructing the data back for visualization

reconst_red = (transform_test_red@PC_red).reshape(n, h, w)
reconst_blue = (transform_test_blue@PC_blue).reshape(n, h, w)
reconst_green = (transform_test_green@PC_green).reshape(n, h, w)

images_test_merged = np.stack((reconst_red, reconst_green, reconst_blue), axis=-1)
plot_image_grid(images_test_merged, "Merged test images after PCA")
