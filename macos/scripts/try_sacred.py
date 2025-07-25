from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver

ex = Experiment()
# ex.observers.append(FileStorageObserver('my_runs'))
ex.observers.append(
    MongoObserver(db_name="sacred_test", url="mongodb://localhost:27017/")
)


@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient


@ex.automain
def my_main(message):
    """See https://sacred.readthedocs.io/en/stable/quickstart.html"""
    print(message)


from sacred import Experiment

ex = Experiment("hello_config")
