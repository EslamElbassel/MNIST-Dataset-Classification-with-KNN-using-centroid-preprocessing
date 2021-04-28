from mlxtend.data import loadlocal_mnist
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def apply_centroid(img):
    feature_vector = []
    Xc = 0
    Yc = 0
    value = 0
    Xsum = 0
    Ysum = 0
    for x in range(0, len(img)):
        for y in range(0, len(img)):
            if img[x][y] != 0:
                value += img[x][y]
                Xsum += x * img[x][y]
                Ysum += y * img[x][y]
    if value != 0:
        Xc = Xsum / value
        Yc = Ysum / value
    feature_vector.append((Xc, Yc))
    return feature_vector


def split_image(img, rows, columns):
    x, y = img.shape
    return (img.reshape(y // rows, rows, -1, columns)
            .swapaxes(1, 2)
            .reshape(-1, rows, columns))


def main():
    (train_images, train_labels) = loadlocal_mnist(images_path='train-images.idx3-ubyte',
                                                   labels_path='train-labels.idx1-ubyte')

    (test_images, test_labels) = loadlocal_mnist(images_path='t10k-images.idx3-ubyte',
                                                 labels_path='t10k-labels.idx1-ubyte')
    train_images = train_images.reshape(len(train_images), 28, 28)

    test_images = test_images.reshape(len(test_images), 28, 28)

    print("train_images ", train_images.shape)
    print("test_images ", test_images.shape)

    train_features = []
    test_features = []

    for image in train_images:
        feature_vector = []
        for window in split_image(image, 7, 7):
            out = apply_centroid(window)
            feature_vector.append(out)
        train_features.append(feature_vector)
    for image in test_images:
        feature_vector = []
        for window in split_image(image, 7, 7):
            out = apply_centroid(window)
            feature_vector.append(out)
        test_features.append(feature_vector)

    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_features = train_features.reshape(60000, 32)
    test_features = test_features.reshape(10000, 32)

    knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_model.fit(train_features, train_labels)
    knn_prediction = knn_model.predict(test_features)
    print("Accuracy = ", accuracy_score(test_labels, knn_prediction) * 100, "%")

if __name__ == '__main__':
    main()
