from config_0 import LitClassifier
from pytorch_lightning.utilities.cli import LightningCLI


def cli_main():
    cli = LightningCLI(LitClassifier)
    cli.trainer.test(cli.model)


# In[ ]:

if __name__ == '__main__':
    cli_main()
