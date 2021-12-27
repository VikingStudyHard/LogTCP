from configparser import RawConfigParser
import sys

sys.path.append('..')


class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = RawConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config

    @property
    def fasttext_model(self):
        return self._config.get("Fasttext", "fasttext_model")

    @property
    def fasttext_dim(self):
        return self._config.getint("Fasttext", "fasttext_dim")

    @property
    def fasttext_thread(self):
        return self._config.getint("Fasttext", "fasttext_thread")

    @property
    def fasttext_lr(self):
        return self._config.getfloat("Fasttext", "fasttext_lr")

    @property
    def fasttext_epoch(self):
        return self._config.getint("Fasttext", "fasttext_epoch")

    @property
    def fasttext_word_ngrams(self):
        return self._config.getint("Fasttext", "fasttext_word_ngrams")

