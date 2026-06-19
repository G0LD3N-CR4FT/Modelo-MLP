import os
import time
from cassandra.cluster import Cluster

try:
    from cassandra.errors import NoHostAvailable
except ModuleNotFoundError:
    try:
        from cassandra import NoHostAvailable
    except ImportError:
        # Fallback genérico caso o pacote esteja corrompido
        class NoHostAvailable(Exception): pass

class CassandraDB:
    def __init__(self):
        self.host = os.getenv("CASSANDRA_HOST", "cassandra")
        self.session = None
        self.cluster = None
        self.connect()

    def connect(self):
        print(f"🔄 Conectando ao cluster Cassandra em: {self.host}...")
        
        tentativas = 30
        for i in range(tentativas):
            try:
                self.cluster = Cluster([self.host], port=9042)

                self.cluster.max_requests_per_connection_local = 2048
                self.cluster.max_requests_per_connection_remote = 2048

                self.session = self.cluster.connect()

                self.session.default_timeout = 60.0

                print("⚡ Conexão de rede estabelecida com o Cassandra!")
                break  # Conectou com sucesso, sai do loop!
            except Exception as e:
                print(f"⏳ Cassandra ainda está inicializando (Erro: {type(e).__name__})... Tentativa {i+1}/{tentativas}. Aguardando 5s...")
                if self.cluster:
                    try: self.cluster.shutdown() 
                    except: pass
                time.sleep(5)
        else:
            raise Exception("❌ Não foi possível conectar ao Cassandra após várias tentativas.")
        
        self.setup_schema()

    def setup_schema(self):
        self.session.execute("""
            CREATE KEYSPACE IF NOT EXISTS mlp_mnist
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};
        """)
        self.session.set_keyspace("mlp_mnist")

        self.session.execute("""
            CREATE TABLE IF NOT EXISTS mnist_dataset (
                split text,
                sample_id int,
                label int,
                pixels list<float>,
                PRIMARY KEY (split, sample_id)
            );
        """)
        print("✅ Keyspace e Tabela mnist_dataset verificados/criados com sucesso!")

    def get_session(self):
        return self.session

    def close(self):
        if self.cluster:
            self.cluster.shutdown()