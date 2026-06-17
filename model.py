import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.special import logsumexp
from tensorflow.keras.datasets import mnist


class MLP():
    def __init__(self, din, dout):
        self.W = (2 * np.random.rand(dout, din) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.b = (2 * np.random.rand(dout) - 1) * (np.sqrt(6) / np.sqrt(din + dout))

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, gradout):
        self.deltaW = gradout.T @ self.x
        self.deltab = gradout.sum(0)
        return gradout @ self.W


class SequentialNN():
    def __init__(self, blocks: list):
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, gradout):
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
        return gradout


class ReLU():
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, gradout):
        new_grad = gradout.copy()
        new_grad[self.x < 0] = 0.
        return new_grad


class LogSoftmax():
    def forward(self, x):
        self.x = x
        return x - logsumexp(x, axis=1)[..., None]

    def backward(self, gradout):
        gradients = np.eye(self.x.shape[1])[None, ...]
        gradients = gradients - (np.exp(self.x) / np.sum(np.exp(self.x), axis=1)[..., None])[..., None]
        return (np.matmul(gradients, gradout[..., None]))[:, :, 0]


class NLLLoss():
    def forward(self, pred, true):
        self.pred = pred
        self.true = true
        loss = 0
        for b in range(pred.shape[0]):
            loss -= pred[b, true[b]]
        return loss

    def backward(self):
        din = self.pred.shape[1]
        jacobian = np.zeros((self.pred.shape[0], din))
        for b in range(self.pred.shape[0]):
            jacobian[b, self.true[b]] = -1
        return jacobian

    def __call__(self, pred, true):
        return self.forward(pred, true)


class Optimizer():
    def __init__(self, lr, compound_nn: SequentialNN):
        self.lr = lr
        self.compound_nn = compound_nn

    def step(self):
        for block in self.compound_nn.blocks:
            if isinstance(block, MLP):
                block.W -= self.lr * block.deltaW
                block.b -= self.lr * block.deltab


def predict(model, image_array):
    return model.forward(image_array.reshape(1, 784)).argmax()


def train(model, optimizer, trainX, trainy, loss_fct=NLLLoss(), nb_epochs=10, batch_size=100):
    for epoch in tqdm(range(nb_epochs)):
        batch_idx = [np.random.randint(0, trainX.shape[0]) for _ in range(batch_size)]
        x = trainX[batch_idx]
        target = trainy[batch_idx]

        prediction = model.forward(x)
        loss_fct(prediction, target)

        gradout = loss_fct.backward()
        model.backward(gradout)
        optimizer.step()


def preprocess_image(file):
    image = Image.open(file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)

    image = image / 255.0
    image = image.reshape(1, 784)

    return image


def load_model():
    # Carregar dados (treino + teste)
    (trainX, trainy), (testX, testy) = mnist.load_data()

    # Normalizar treino
    trainX = (trainX - 127.5) / 127.5
    trainX = trainX.reshape(trainX.shape[0], 28 * 28)

    # Normalizar teste
    testX = (testX - 127.5) / 127.5
    testX = testX.reshape(testX.shape[0], 28 * 28)

    # Criar modelo
    mlp = SequentialNN([
        MLP(28*28, 128), ReLU(),
        MLP(128, 64), ReLU(),
        MLP(64, 10), LogSoftmax()
    ])

    # Otimizador
    optimizer = Optimizer(1e-3, mlp)

    # Treinar
    train(mlp, optimizer, trainX, trainy, nb_epochs=10)

    # =========================
    # Calcular acurácia (rápido)
    # =========================
    preds = mlp.forward(testX).argmax(axis=1)
    accuracy = (preds == testy).mean()

    print('Test accuracy:', accuracy * 100, '%')

    return mlp