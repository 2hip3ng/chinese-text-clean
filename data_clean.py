# coding:utf-8

from __future__ import absolute_import, division, print_function

import argparse
import sys
import csv
import logging
import os

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for data clean."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):
    """Base class for read data for data clean"""
    def get_examples(self, input_file):
        raise NotImplementedError

    def write_examples(self, examples, output_file):
        raise NotImplementedError

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DemoProcessor(DataProcessor):
    """Processor for the Demo data set."""

    def get_examples(self, input_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(input_file))
        return self._create_examples(
            self._read_file(input_file), "data")

    def write_examples(self, examples, output_file):
        logger.info("WRITING TO OUTPUT FILE {}".format(output_file))
        with open(output_file, 'w', encoding='utf-8') as writer:
            for example in examples:
                writer.write(example.label + '\t' + example.text_a + '\n')

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = None
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

#  去URL
def rmURL(line):
    import re
    pattern = re.compile(r'http[a-zA-Z0-9.?/&=:]*')
    return pattern.sub('', line.strip())


# 去非中文、英文、数字字符， 如表情，小语种字符
def rmUNK(line):
    import re
    pattern = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+', re.UNICODE)
    return pattern.sub('', line.strip())



# 去停用词
def rmStopwords(line, stopwords):
    out = []
    for word in line:
        if word not in stopwords:
            out.append(word)
    return out



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--input_file',
        default=None,
        type=str,
        required=True,
        help='The input file which need clean.')

    parser.add_argument('--output_file',
        default=None,
        type=str,
        required=True,
        help='The output file of cleaned data')

    parser.add_argument('--task_name',
        default=None,
        type=str,
        required=True,
        choices=['demo'],
        help='The name of the task to clean.')


    # Optional parameters
    parser.add_argument('--rm_url',
        default=True,
        type=bool,
        help='whether to remove url link.')

    parser.add_argument('--rm_unknown_char',
        default=True,
        type=bool,
        help='whether to remove unknown char')

    parser.add_argument('--jieba_cut',
        default=True,
        type=bool,
        help='whether to use jieba to cut the sentence.')

    parser.add_argument('--jieba_vocab_file',
        default=None,
        type=str,
        help='Custom vocab file for jieba.')

    parser.add_argument('--stopwords_file',
        default=None,
        type=str,
        help='whether to remove the stopwords and stopwords vocab file.')

    args = parser.parse_args()

    processors = {
        'demo': DemoProcessor,
        # add in here your processor, key: should be lower case
    }

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    if not os.path.exists(args.input_file):
        raise ValueError('input_file {} does not exist, should be a data file which need to be cleaned.'.format(args.input_file))
    if os.path.exists(args.output_file):
        raise ValueError('input_file {} has existed, should be a new file.'.format(args.output_file))

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task nor found: %s' % (task_name))

    processor = processors[task_name]()
    data_examples = processor.get_examples(args.input_file)
    # data_examples  [inputexample1, inputexample2, inputexample3,  ...]

    if args.rm_url:
        logging.info('starting remove url.')
        url_examples = data_examples
        data_examples = []
        for url_example in url_examples:
            guid = url_example.guid
            text_a = url_example.text_a
            text_b = url_example.text_b
            label = url_example.label

            text_a = rmURL(text_a)
            if text_b is not None:
                text_b = rmURL(text_b)
            example = InputExample(guid, text_a, text_b, label)
            data_examples.append(example)
        logging.info('finishing remove url.')


    if args.rm_unknown_char:
        logging.info('starting remove unknown char.')
        unknown_char_examples = data_examples
        data_examples = []
        for unknown_char_example in unknown_char_examples:
            guid = unknown_char_example.guid
            text_a = unknown_char_example.text_a
            text_b = unknown_char_example.text_b
            label = unknown_char_example.label

            text_a = rmUNK(text_a)
            if text_b is not None:
                text_b = rmUNK(text_b)
            example = InputExample(guid, text_a, text_b, label)
            data_examples.append(example)

        logging.info('finishing remove unknown char.')


    if args.jieba_cut:
        logging.info('starting tokenization.')
        import jieba
        if args.jieba_vocab_file:
            jieba.load_userdict(args.jieba_vocab_file)

        jieba_examples = data_examples
        data_examples = []
        for jieba_example in jieba_examples:
            guid = jieba_example.guid
            text_a = jieba_example.text_a
            text_b = jieba_example.text_b
            label = jieba_example.label

            text_a = list(jieba.cut(text_a.strip()))
            if text_b is not None:
                text_b = list(jieba.cut(text_b.strip()))
            example = InputExample(guid, text_a, text_b, label)
            data_examples.append(example)
        logging.info('finishing tokenization.')


    if args.stopwords_file:
        logging.info('starting remove stopwords.')
        stopwords = []
        with open(args.stopwords_file, 'r') as f:
            for line in f.readlines():
                stopwords.append(line.strip())

        stopwords_examples = data_examples
        data_examples = []
        for stopwords_example in stopwords_examples:
            guid = stopwords_example.guid
            text_a = stopwords_example.text_a
            text_b = stopwords_example.text_b
            label = stopwords_example.label

            text_a = rmStopwords(text_a, stopwords)
            if text_b is not None:
                text_b = rmStopwords(text_b, stopwords)
            example = InputExample(guid, ' '.join(text_a), text_b, label)
            data_examples.append(example)
        logging.info('finishing remove stopwords.')

    # 做一些其他清理工作
    logging.info('starting other clean operation.')
    otherOP_examples = data_examples
    data_examples = []
    for otherOP_example in otherOP_examples:
        guid = otherOP_example.guid
        text_a = otherOP_example.text_a
        text_b = otherOP_example.text_b
        label = otherOP_example.label

        # text_a = text_a.replace('\n', '').replace('\r', '')
        if text_b is not None:
            pass
            # text_b = text_b.replace('\n', '').replace('\r', '')
        example = InputExample(guid, text_a, text_b, label)
        data_examples.append(example)

    logging.info('finishing other clean operation.')


    processor.write_examples(data_examples, args.output_file)
    logging.info('Data clean finished!')

if __name__ == '__main__':
    main()
