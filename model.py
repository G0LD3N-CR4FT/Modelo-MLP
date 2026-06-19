import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.special import logsumexp
from db.database import CassandraDB


class MLP():
    def __init__(self, din, dout):
        # Inicialização Xavier/Glorot
        self.W = (2 * np.random.rand(dout, din) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.b = np.zeros(dout)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, gradout):
        # Média sobre o batch
        self.deltaW = gradout.T @ self.x / gradout.shape[0]
        self.deltab = gradout.sum(0)     / gradout.shape[0]
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
        self.probs = np.exp(x - logsumexp(x, axis=1)[..., None])
        return np.log(self.probs + 1e-15) # Evita log(0)

    def backward(self, gradout):
        return gradout - self.probs * gradout.sum(axis=1)[..., None]


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
        batch_size = self.pred.shape[0]  # <--- Captura dinamicamente o tamanho do lote
        jacobian = np.zeros((batch_size, din))
        for b in range(batch_size):
            jacobian[b, self.true[b]] = -1
        return jacobian / batch_size

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
    num_samples = trainX.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(nb_epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        running_loss = 0
        # tqdm agora envolve os lotes internos, não as épocas externas
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{nb_epochs}") as pbar:
            for b in range(num_batches):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                x = trainX[batch_idx]
                target = trainy[batch_idx]

                prediction = model.forward(x)
                loss = loss_fct(prediction, target)
                running_loss += loss

                gradout = loss_fct.backward()
                model.backward(gradout)
                optimizer.step()
                
                pbar.set_postfix({'loss': running_loss / (b + 1)})
                pbar.update(1)


def preprocess_image(file):
    image = Image.open(file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    # Corrigido para bater com a normalização (-1 a 1) feita no treino
    image = (image - 127.5) / 127.5
    image = image.reshape(1, 784)
    return image


def load_data_from_cassandra():
    """Busca o dataset MNIST de dentro do Cassandra e reconstrói as matrizes NumPy"""
    print("📥 Puxando dados de treino do Apache Cassandra para a MLP...")
    db = CassandraDB()
    session = db.get_session()
    
    # 1. Buscar dados de Treino
    query_train = "SELECT label, pixels FROM mnist_dataset WHERE split = 'train'"
    results_train = session.execute(query_train)
    
    train_x_list = []
    train_y_list = []
    
    for row in results_train:
        train_x_list.append(row.pixels)
        train_y_list.append(row.label)
        
    # 2. Buscar dados de Teste
    print("📥 Puxando dados de teste do Apache Cassandra para validação...")
    query_test = "SELECT label, pixels FROM mnist_dataset WHERE split = 'test'"
    results_test = session.execute(query_test)
    
    test_x_list = []
    test_y_list = []
    
    for row in results_test:
        test_x_list.append(row.pixels)
        test_y_list.append(row.label)

    # 3. Converter de volta para os arrays de alta performance do NumPy
    trainX = np.array(train_x_list, dtype=np.float32)
    trainy = np.array(train_y_list, dtype=np.int64)
    testX = np.array(test_x_list, dtype=np.float32)
    testy = np.array(test_y_list, dtype=np.int64)
    
    print(f"✅ Dataset carregado com sucesso do Banco! Treino: {trainX.shape}, Teste: {testX.shape}")
    return trainX, trainy, testX, testy

def load_model():
    MODEL_PATH = "mlp_mnist_model.pkl"
    
    # Se o modelo já foi treinado antes, carrega ele instantaneamente do arquivo binário
    if os.path.exists(MODEL_PATH):
        print("💾 Carregando pesos e bias pré-treinados do arquivo local...")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
            
    # Se não existir, extrai do Cassandra e treina a rede
    print("🧠 Modelo treinado não encontrado localmente.")
    trainX, trainy, testX, testy = load_data_from_cassandra()

    # Criar modelo com a sua arquitetura original
    mlp = SequentialNN([
        MLP(28*28, 128), ReLU(),
        MLP(128, 64), ReLU(),
        MLP(64, 10), LogSoftmax()
    ])

    # Otimizador
    optimizer = Optimizer(1e-1, mlp)

    # Treinar usando os dados do Cassandra
    train(mlp, optimizer, trainX, trainy, nb_epochs=10)

    # Calcular acurácia no bloco de teste extraído do Cassandra
    preds = mlp.forward(testX).argmax(axis=1)
    accuracy = (preds == testy).mean()
    print('Test accuracy:', accuracy * 100, '%')

    # Salva o arquivo em disco para os próximos boots do contêiner
    print(f"💾 Salvando neurônios treinados em '{MODEL_PATH}'...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(mlp, f)

    return mlp