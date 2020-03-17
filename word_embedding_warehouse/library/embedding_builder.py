__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'

# libraries
from word_embedding_warehouse.library.files_and_folders import *
from word_embedding_warehouse.library.downloading import *
from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert
import torch, json


class WordEmbeddingWarehouseBuilder:
    """
    Word Embedding Warehouse Builder
    ==========
    This class provides a considerably user-friendly interface for the user to fetch, build, convert to pytorch weights, and
    process almost every well-known word-embedding, in all their variants.

    This class is the main agent to be used for word-embedding warehouse building. To use,
    instantiate it and run the `build_warehouse` method with no arguments.
    """

    def __init__(
            self,
            output_path: str = None,
            download_links_file: str = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                    '../configuration.json')),
            clear_cache: bool = False
    ):
        """
        This serves as the constructor of this class.
        Note that this class uses the files AS IS therefore it is possible for certain alterations to be necessary
        in order to get them to work, for example BioBERT files might be either in zip format or something else, etc.

        Parameters
        ----------
        output_path: `str`, optional (default=None)
            The path to the output warehouse, which is going to be a folder preferably
        download_links_file: `str`, optional (default=`os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/miscellaneous/token_embedding_download_links.json'))`
            The path to the JSON file including the download links to all the embeddings.
        clear_cache: `bool`, optional (default=False)
            This optional variable helps if you're running short on storage, and setting it to true enforces
            fast removal of temporary files and compressed archives after usage.
        """
        # the output_path is required, therefore, it should be given.
        # sanity check:
        assert output_path is not None, "Please provide the output path in which the warehouse is supposed to be built."

        # the path to the output folder will be stored in the class:
        self.output_path = os.path.abspath(output_path)

        # the `download_links_file` is to be parsed:
        # with open(download_links_file) as handle:
        #     self.download_links = json.load(handle)
        self.download_links = {
            "token_embedding_download_links": {
                "ncbibert_pubmedmimic_small": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12.zip",
                "ncbibert_pubmedmimic_large": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-24_H-1024_A-16.zip",
                "ncbibert_pubmed_small": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_uncased_L-12_H-768_A-12.zip",
                "ncbibert_pubmed_large": "https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_uncased_L-24_H-1024_A-16.zip",
                "biobert_v1.0_pmc": "https://github.com/naver/biobert-pretrained/releases/download/v1.0-pmc/biobert_v1.0_pmc.tar.gz",
                "biobert_v1.0_pubmedpmc": "https://github.com/naver/biobert-pretrained/releases/download/v1.0-pubmed-pmc/biobert_v1.0_pubmed_pmc.tar.gz",
                "biobert_v1.1_pubmed": "https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz",
                "bert_small": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
                "bert_large": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
                "elmo_small": {
                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
                },
                "elmo_medium": {
                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5",
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
                },
                "elmo_original": {
                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                },
                "elmo_original_5b": {
                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
                },
                "elmo_pubmed": {
                    "weights": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5",
                    "options": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
                },
                "glove_files": {
                    "glove.twitter.27B.txt": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
                    "glove.840B.300d.txt": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
                    "glove.42B.300d.txt": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
                    "glove.6B.txt": "http://nlp.stanford.edu/data/glove.6B.zip"
                }
            }
        }
        self.download_links = self.download_links['token_embedding_download_links']

        # forcing the temporary folder (which will be used) to exist also forces its parent to exist.
        self.temp_path = force_folder_to_exist(os.path.join(self.output_path, 'tmp'))
        self.clear_cache = clear_cache


def fetch_ncbibert_files(self):
    """
    Embedding Family Fetcher Method
    """
    embeddings = [
        'ncbibert_pubmedmimic_small',
        'ncbibert_pubmedmimic_large',
        'ncbibert_pubmed_small',
        'ncbibert_pubmed_large'
    ]

    for embedding in embeddings:
        download_file_to_path(
            file_link=self.download_links[embedding],
            path=self.temp_path,
            output_file_name=embedding + '.' + self.download_links[embedding].split('.')[-1]
        )


def fetch_biobert_files(self):
    """
    Embedding Family Fetcher Method
    """
    embeddings = [
        'biobert_v1.0_pmc',
        'biobert_v1.0_pubmedpmc',
        'biobert_v1.1_pubmed'
    ]

    for embedding in embeddings:
        download_file_to_path(
            file_link=self.download_links[embedding],
            path=self.temp_path,
            output_file_name=embedding + '.' + self.download_links[embedding].split('.')[-1]
        )


def fetch_bert_files(self):
    """
    Embedding Family Fetcher Method
    """
    embeddings = [
        'bert_small',
        'bert_large'
    ]

    for embedding in embeddings:
        download_file_to_path(
            file_link=self.download_links[embedding],
            path=self.temp_path,
            output_file_name=embedding + '.' + self.download_links[embedding].split('.')[-1]
        )


def fetch_elmo_files(self):
    """
    Embedding Family Fetcher Method
    """
    embeddings = [
        'elmo_small',
        'elmo_medium',
        'elmo_original',
        'elmo_original_5b',
        'elmo_pubmed'
    ]

    for embedding in embeddings:
        download_file_to_path(
            file_link=self.download_links[embedding]['weights'],
            path=self.temp_path,
            output_file_name=embedding + '.hdf5'
        )
        download_file_to_path(
            file_link=self.download_links[embedding]['options'],
            path=self.temp_path,
            output_file_name=embedding + '.json'
        )


def fetch_glove_files(self):
    """
    Embedding Family Fetcher Method
    """
    glove_download_links = self.download_links['glove_files']
    for file in glove_download_links:
        download_file_to_path(
            file_link=glove_download_links[file],
            path=self.temp_path,
            output_file_name=file
        )


def fetch_all_models_for_all_embeddings(self):
    """
    Fetching all of the model families can take place by using :meth:`fetch_all_models_for_all_embeddings`.
    """
    self.fetch_ncbibert_files()
    self.fetch_biobert_files()
    self.fetch_bert_files()
    self.fetch_elmo_files()
    self.fetch_glove_files()


def build_ncbibert_warehouse(self):
    """
    Embedding family builder
    """
    embeddings = [
        'ncbibert_pubmedmimic_small',
        'ncbibert_pubmedmimic_large',
        'ncbibert_pubmed_small',
        'ncbibert_pubmed_large'
    ]
    command_set = []

    for embedding in embeddings:
        force_folder_to_exist(os.path.join(self.output_path, 'ncbibert', embedding))
        os.system(
            'unzip ' + os.path.join(self.temp_path, embedding + '.zip') + ' -d ' + os.path.join(self.output_path,
                                                                                                'ncbibert',
                                                                                                embedding)
        )
        if self.clear_cache:
            os.system(
                'rm ' + os.path.join(self.temp_path, embedding + '.zip')
            )


def build_bert_warehouse(self):
    """
    Embedding family builder
    """
    embeddings = [
        'bert_small',
        'bert_large'
    ]
    command_set = []

    for embedding in embeddings:
        force_folder_to_exist(os.path.join(self.output_path, 'bert', embedding))
        os.system(
            'unzip ' + os.path.join(self.temp_path, embedding + '.zip') + ' -d ' + os.path.join(self.output_path,
                                                                                                'bert',
                                                                                                embedding)
        )

        the_folder_name = os.listdir(os.path.join(self.output_path, 'bert', embedding))[0]

        os.system(
            'mv ' + os.path.join(self.output_path, 'bert', embedding, the_folder_name, '*') + ' ' + os.path.join(
                self.output_path, 'bert', embedding)
        )

        if self.clear_cache:
            os.system(
                'rm ' + os.path.join(self.temp_path, embedding + '.zip')
            )


def build_biobert_warehouse(self):
    """
    Embedding family builder
    """
    embeddings = [
        'biobert_v1.0_pmc',
        'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed'
    ]

    command_set = []

    for embedding in embeddings:
        force_folder_to_exist(os.path.join(self.output_path, 'biobert', embedding))

        # embedding names and folders for biobert are similar, so there is no harm in doing this.
        os.system(
            'tar xvf ' + os.path.join(self.temp_path, embedding + '.gz') + ' -C ' + os.path.join(self.output_path,
                                                                                                 'biobert')
        )
        if self.clear_cache:
            os.system(
                'rm ' + os.path.join(self.temp_path, embedding + '.gz')
            )

    os.system(
        'mv ' + os.path.join(self.output_path, 'biobert', 'biobert_v1.1_pubmed',
                             'model.ckpt-1000000.data-00000-of-00001') + ' ' + os.path.join(self.output_path,
                                                                                            'biobert',
                                                                                            'biobert_v1.1_pubmed',
                                                                                            'bert_model.ckpt.data-00000-of-00001')
    )
    os.system(
        'mv ' + os.path.join(self.output_path,
                             'biobert', 'biobert_v1.1_pubmed', 'model.ckpt-1000000.index') + ' ' + os.path.join(
            self.output_path,
            'biobert', 'biobert_v1.1_pubmed', 'bert_model.ckpt.index')
    )
    os.system(
        'mv ' + os.path.join(self.output_path,
                             'biobert') + 'model.ckpt-1000000.meta ' + os.path.join(self.output_path,
                                                                                    'biobert',
                                                                                    'biobert_v1.1_pubmed',
                                                                                    'bert_model.ckpt.meta')
    )


def build_elmo_warehouse(self):
    """
    Embedding family builder
    """
    embeddings = [
        'elmo_small',
        'elmo_medium',
        'elmo_original',
        'elmo_original_5b',
        'elmo_pubmed'
    ]

    command_set = []

    for embedding in embeddings:
        force_folder_to_exist(os.path.join(self.output_path, 'elmo', embedding))
        os.system(
            'mv ' + os.path.join(self.temp_path, embedding + '*') + ' ' + os.path.join(self.output_path, 'elmo',
                                                                                       embedding)
        )


def build_glove_warehouse(self):
    """
    Embedding family builder
    """
    embeddings = list(self.download_links['glove_files'].keys())
    command_set = []
    force_folder_to_exist(os.path.join(self.output_path, 'glove'))
    for embedding in embeddings:
        os.system(
            'mv ' + os.path.join(self.temp_path, embedding) + ' ' + os.path.join(self.output_path, 'glove')
        )


def build_warehouse(self) -> None:
    """
    Embedding warehouse builder for all families and all variants that are specified in this platform.
    """
    # building the warehouses
    self.build_ncbibert_warehouse()
    self.build_biobert_warehouse()
    self.build_bert_warehouse()
    self.build_elmo_warehouse()
    self.build_glove_warehouse()

    output_folder = self.output_path

    force_folder_to_exist(os.path.join(output_folder, 'bert/bert'))
    os.system(
        'mv ' + os.path.join(output_folder, 'bert/bert_small') + ' ' + os.path.join(output_folder, 'bert/bert'))
    os.system(
        'mv ' + os.path.join(output_folder, 'bert/bert_large') + ' ' + os.path.join(output_folder, 'bert/bert'))
    os.system('mv ' + os.path.join(output_folder, 'biobert') + ' ' + os.path.join(output_folder, 'bert'))
    os.system('mv ' + os.path.join(output_folder, 'ncbibert') + ' ' + os.path.join(output_folder, 'bert'))


def convert_all_tensorflow_bert_weights_to_pytorch(self, input_folder: str) -> None:
    """
    Tensorflow to Pytorch weight conversion based on huggingface's library

    Parameters
    ----------
    input_folder: `str`, required
        The folder containing the tensorflow files
    """
    files = [e for e in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, e))]
    folders = [os.path.join(input_folder, e) for e in os.listdir(input_folder) if
               os.path.isdir(os.path.join(input_folder, e))]

    flag = -4
    for file in files:
        if file == 'vocab.txt' or \
                file.endswith('.data-00000-of-00001') or \
                file.endswith('.index') or \
                file.endswith('.meta') or \
                file.endswith('.json'):
            flag += 1
            if file.endswith('.json'):
                config_file = file

    if flag > 0:
        assert type(config_file) == str, "no valid config file, but is attempting to convert"
        pytorch_path = os.path.join(input_folder, 'pytorch')
        tensorflow_path = os.path.join(input_folder, 'tensorflow')

        force_folder_to_exist(pytorch_path)
        force_folder_to_exist(tensorflow_path)

        os.system('mv ' + os.path.join(input_folder, '*.*') + ' ' + tensorflow_path)
        os.system('cp ' + os.path.join(tensorflow_path, '*.txt') + ' ' + pytorch_path)
        os.system('cp ' + os.path.join(tensorflow_path, '*.json') + ' ' + pytorch_path)

        config = BertConfig.from_json_file(os.path.join(tensorflow_path, config_file))
        model = BertForPreTraining(config)
        load_tf_weights_in_bert(model=model, tf_checkpoint_path=os.path.join(tensorflow_path, 'bert_model.ckpt'))
        torch.save(model.state_dict(), os.path.join(pytorch_path, 'pytorch_model.bin'))

    else:
        for folder in folders:
            self.convert_all_tensorflow_bert_weights_to_pytorch(input_folder=folder)
