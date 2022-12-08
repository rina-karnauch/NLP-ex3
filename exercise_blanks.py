import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    avg_embedding = np.zeros(embedding_dim)
    empty_vector = np.zeros(embedding_dim)
    sentence = sent.text
    for word in sentence:
        word_vec = word_to_vec.get(word, empty_vector)
        avg_embedding += word_vec
    avg_embedding = avg_embedding / len(sentence)
    return avg_embedding


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot_vec = np.zeros(size)
    one_hot_vec[ind] = 1
    return one_hot_vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    V_size = len(word_to_ind)
    sent_vec = np.zeros(V_size)
    sentence = sent.text
    sentence_size = len(sentence)
    for word in sentence:
        ind = word_to_ind[word]
        sent_vec[ind] += 1
    average = sent_vec / sentence_size
    return average


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_to_index_dict = dict()
    counter = 0
    for word in words_list:
        if word not in word_to_index_dict.keys():
            word_to_index_dict[word] = counter
            counter += 1
    return word_to_index_dict


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    zero_vector = np.zeros(embedding_dim)
    sent_to_word_embeddings = []
    sentence = sent.text

    if len(sentence) > seq_len:
        sentence = sentence[:seq_len]

    for word in sentence:
        vec = np.array(word_to_vec.get(word, zero_vector))
        sent_to_word_embeddings.append(vec)

    while len(sent_to_word_embeddings) < seq_len:
        sent_to_word_embeddings.append(zero_vector)
    return np.array(sent_to_word_embeddings)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self._output_dim = 1
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._dropout = dropout

        self._lstm = nn.LSTM(self._embedding_dim, self._hidden_dim, self._n_layers, bidirectional=True,
                             dropout=self._dropout, batch_first=True)
        self._nn = nn.Linear(self._hidden_dim * 2, self._output_dim)
        self._linear = None
        self._criterion = None

    def forward(self, text):
        lstm_out, (lstm_h, lstm_c) = self._lstm(text.float())
        h_n, h_0_r = lstm_h[0], lstm_h[1]
        concat_hiddens = torch.cat((h_n, h_0_r), dim=1)
        self._linear = self._nn(concat_hiddens)
        return self._linear

    def predict(self, text):
        self.forward(text)
        self._output = torch.sigmoid(self._linear)
        return self._output

    def get_criterion(self):
        return self._criterion


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self._in_features = embedding_dim
        self._out_features = 1
        self._nn = nn.Linear(self._in_features, self._out_features)
        self._linear = None
        self._output = None
        self._criterion = None

    def forward(self, x):
        x = x.type(torch.float32)
        self._linear = self._nn(x)
        return self._linear

    def predict(self, x):
        self.forward(x)
        self._output = torch.sigmoid(self._linear)
        return self._output

    def get_criterion(self):
        return self._criterion


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    rounded_preds = [round(p) for p in preds.numpy()]

    count_sames = 0
    for i in range(len(rounded_preds)):
        if rounded_preds[i] == y[i]:
            count_sames += 1

    accuracy = count_sames / len(preds)
    return accuracy


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    amount, accuracy, model_loss = 0, 0, 0

    for x, y in data_iterator:
        optimizer.zero_grad()
        pred = torch.flatten(model.forward(x))
        pred_sigmoid = torch.sigmoid(pred)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        embedded_length = len(x)

        amount += embedded_length
        accuracy += embedded_length * binary_accuracy(pred_sigmoid.detach(), y.detach())
        model_loss += float(loss) * embedded_length

    return model_loss / amount, accuracy / amount


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    amount, accuracy, model_loss = 0, 0, 0

    for x, y in data_iterator:
        pred_sigmoid = torch.flatten(model.predict(x))
        loss = criterion(pred_sigmoid, y)

        embedded_length = len(x)

        amount += embedded_length
        accuracy += embedded_length * binary_accuracy(pred_sigmoid.detach(), y.detach())
        model_loss += float(loss) * embedded_length

    return model_loss / amount, accuracy / amount


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = np.array([])
    for x, y in data_iter:
        pred = torch.flatten(model.predict(x)).detach()
        predictions = np.append(predictions, pred)
    return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """

    train_loss_arr, train_accuracy_arr, validation_loss_arr, validation_accuracy_arr = [], [], [], []
    data_iterator_train = data_manager.get_torch_iterator(TRAIN)
    data_iterator_validation = data_manager.get_torch_iterator(VAL)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    model._criterion = criterion

    for n in tqdm.tqdm(list(range(n_epochs))):
        train_loss, train_accuracy = train_epoch(model, data_iterator_train, optimizer, criterion)
        test_loss, test_accuracy = evaluate(model, data_iterator_validation, criterion)

        train_loss_arr.append(train_loss)
        train_accuracy_arr.append(train_accuracy)
        validation_loss_arr.append(test_loss)
        validation_accuracy_arr.append(test_accuracy)
    return train_loss_arr, train_accuracy_arr, validation_loss_arr, validation_accuracy_arr


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """

    lr = 0.01
    n_epochs = 20
    batch_size = 64
    weight_decay = 0.001

    data_manager = DataManager(batch_size=batch_size)
    input_shape = len(data_manager.sentiment_dataset.get_word_counts())
    model = LogLinear(input_shape)

    # ---------- plotting ----------
    train_loss_arr, train_accuracy_arr, validation_loss_arr, validation_accuracy_arr = train_model(model, data_manager,
                                                                                                   n_epochs,
                                                                                                   lr, weight_decay)

    # ---------- loss plots ----------
    plt.figure()
    plt.plot(train_loss_arr, label="train loss", color="firebrick")
    plt.plot(validation_loss_arr, label="validation loss", color="rosybrown")
    plt.legend()
    plt.title("training-validation loss with OH")
    plt.show()

    # ---------- accuracy plots ----------
    plt.figure()
    plt.plot(train_accuracy_arr, label="train accuracy", color="steelblue")
    plt.plot(validation_accuracy_arr, label="validation accuracy", color="skyblue")
    plt.title("training-validation accuracy with OH")
    plt.legend()
    plt.show()

    # ---------- test model ----------
    test_model(model, data_manager, model.get_criterion(), "6")


def test_model(model, data_manager, criterion, question_number):
    data_iterator_test = data_manager.get_torch_iterator(TEST)
    y_data_set = [d[1] for d in data_iterator_test.dataset]

    test_loss, test_accuracy = evaluate(model, data_iterator_test, criterion)
    predictions_test = np.round(get_predictions_for_data(model, data_iterator_test))

    print("Q{q}a: test loss: {test_loss}".format(test_loss=test_loss, q=question_number))
    print("Q{q}a: test accuracy {test_accuracy}: ".format(test_accuracy=test_accuracy, q=question_number))

    print(" -------------------------------------- ")

    dataset = data_loader.SentimentTreeBank()
    test_sentences = dataset.get_test_set()
    negated_polarity_indices = data_loader.get_negated_polarity_examples(test_sentences)
    rare_words_indices = data_loader.get_rare_words_examples(test_sentences, dataset)

    accuracy_negated = 0
    for i in negated_polarity_indices:
        y_i = y_data_set[i]
        if predictions_test[i] == y_i:
            accuracy_negated += 1
    accuracy_rare = 0
    for i in rare_words_indices:
        y_i = y_data_set[i]
        if predictions_test[i] == y_i:
            accuracy_rare += 1

    print("Q{q}c: accuracy over negated words: {accuracy_negative}".format(
        accuracy_negative=accuracy_negated / len(negated_polarity_indices), q=question_number))
    print(
        "Q{q}d: accuracy over rare words: {accuracy_rare}".format(accuracy_rare=accuracy_rare / len(rare_words_indices),
                                                                  q=question_number))

    print(" -------------------------------------- ")


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    lr = 0.01
    n_epochs = 20
    batch_size = 64
    weight_decay = 0.001

    input_shape = W2V_EMBEDDING_DIM
    data_manager = DataManager(batch_size=batch_size, data_type=W2V_AVERAGE, embedding_dim=input_shape)
    model = LogLinear(input_shape)

    # ---------- plotting ----------
    train_loss_arr, train_accuracy_arr, validation_loss_arr, validation_accuracy_arr = train_model(model, data_manager,
                                                                                                   n_epochs,
                                                                                                   lr, weight_decay)

    # ---------- loss plots ----------
    plt.figure()
    plt.plot(train_loss_arr, label="train loss", color="mediumpurple")
    plt.plot(validation_loss_arr, label="validation loss", color="rebeccapurple")
    plt.legend()
    plt.title("training-validation loss with W2V")
    plt.show()

    # ---------- accuracy plots ----------
    plt.figure()
    plt.plot(train_accuracy_arr, label="train accuracy", color="yellowgreen")
    plt.plot(validation_accuracy_arr, label="validation accuracy", color="olivedrab")
    plt.title("training-validation accuracy with W2V")
    plt.legend()
    plt.show()

    # ---------- test model ----------
    test_model(model, data_manager, model.get_criterion(), "7")


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    lr = 0.001
    hidden_dim = 100
    n_layers = 1
    dropout = 0.5
    n_epochs = 4
    batch_size = 64
    weight_decay = 0.0001

    input_shape = W2V_EMBEDDING_DIM
    data_manager = DataManager(batch_size=batch_size, data_type=W2V_SEQUENCE, embedding_dim=input_shape)
    model = LSTM(input_shape, hidden_dim, n_layers, dropout)

    # ---------- plotting ----------
    train_loss_arr, train_accuracy_arr, validation_loss_arr, validation_accuracy_arr = train_model(model, data_manager,
                                                                                                   n_epochs,
                                                                                                   lr, weight_decay)

    # ---------- loss plots ----------
    plt.figure()
    plt.plot(train_loss_arr, label="train loss", color="burlywood")
    plt.plot(validation_loss_arr, label="validation loss", color="darkgoldenrod")
    plt.legend()
    plt.title("training-validation loss with LSTM")
    plt.show()

    # ---------- accuracy plots ----------
    plt.figure()
    plt.plot(train_accuracy_arr, label="train accuracy", color="palevioletred")
    plt.plot(validation_accuracy_arr, label="validation accuracy", color="crimson")
    plt.title("training-validation accuracy with LSTM")
    plt.legend()
    plt.show()

    # ---------- test model ----------
    test_model(model, data_manager, model.get_criterion(), "8")


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    train_log_linear_with_w2v()
    train_lstm_with_w2v()
