import numpy as np
from tensorflow.keras.datasets import mnist
from db.database import CassandraDB
from cassandra.query import SimpleStatement

def seed_mnist_data():
    db = CassandraDB()
    session = db.get_session()

    # Verificação de Idempotência: Checa se o banco já está populado
    print("🔍 Verificando se os dados já existem no Cassandra...")
    row = session.execute("SELECT count(*) FROM mnist_dataset WHERE split='train' LIMIT 1").one()
    if row and row.count > 0:
        print("⏭️ Dados do MNIST já estão no banco de dados. Pulando Seeder.")
        return

    print("📥 Baixando MNIST temporariamente para popular o banco...")
    (trainX, trainy), (testX, testy) = mnist.load_data()

    # Prepara a Query (PreparedStatement) - O Cassandra compila a estrutura uma vez só,
    # tornando a inserção de milhares de linhas infinitamente mais rápida.
    query = """
        INSERT INTO mnist_dataset (split, sample_id, label, pixels)
        VALUES (?, ?, ?, ?)
    """
    prepared = session.prepare(query)

    def insert_split(X, y, split_name):
        print(f"🚀 Injetando dados de '{split_name}' no Cassandra (Total: {len(X)} amostras)...")
        
        futures = []
        MAX_CONCURRENT_REQUESTS = 100
        
        for i in range(len(X)):
            # Normaliza os pixels entre -1 e 1 e converte para lista comum do Python
            pixels_list = ((X[i].flatten() - 127.5) / 127.5).astype(float).tolist()
            label = int(y[i])

            # execute_async envia o dado para o banco e NÃO espera ele responder para enviar o próximo.
            # Disparamos milhares de registros em paralelo!
            future = session.execute_async(prepared, (split_name, i, label, pixels_list))
            futures.append(future)

            if len(futures) >= MAX_CONCURRENT_REQUESTS:
              futures[0].result()  # Bloqueia esperando a primeira da fila concluir
              futures.pop(0)

            # Para não estourar a memória RAM do driver com promessas abertas,
            # limpamos o buffer a cada 5000 envios
            if i % 2000 == 0 and i > 0:
                for f in futures:
                    f.result() # Garante que o bloco terminou de gravar
                futures = []
                print(f"       -> {i} registros processados...")

        # Limpa o restante das promessas que ficaram no final do loop
        for f in futures:
            f.result()

    # Executa a carga para os dois blocos de dados
    insert_split(trainX, trainy, "train")
    insert_split(testX, testy, "test")
    
    print("🎉 Banco de dados Cassandra populado com sucesso com o MNIST completo!")

if __name__ == "__main__":
    seed_mnist_data()