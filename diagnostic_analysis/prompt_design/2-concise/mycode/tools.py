import numpy as np
from collections import defaultdict


def get_datasets(argv):
    print('\nSET UP DATASET\n')
    # dataset: 1D: n_docs, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, response1, ... , label)
    print('\nLoad dataset...')
    train_dataset, words = load_dataset(fn=argv['train_data'], data_size=argv['data_size'])
    dev_dataset, words = load_dataset(fn=argv['dev_data'], vocab=words, data_size=argv['data_size'])
    test_dataset, words = load_dataset(fn=argv['test_data'], vocab=words, data_size=argv['data_size'])
    return train_dataset, dev_dataset, test_dataset, words

import gzip


def load_dataset(fn, vocab=set([]), data_size=1000000, test=False):
    """
    :param fn: file name
    :param vocab: vocab set
    :param data_size: how many threads are used
    :return: threads: 1D: n_threads, 2D: n_utterances, 3D: elem=(time, speaker_id, addressee_id, cand_res1, ... , label)
    """
    if fn is None:
        return None, vocab

    threads = []
    thread = []
    file_open = gzip.open if fn.endswith(".gz") else open

    with file_open(fn, 'r') as gf:
        # line: (time, speaker_id, addressee_id, cand_res1, cand_res2, ... , label)
        for line in gf:
            line = line.decode()
            line = line.split("\t")

            if len(line) < 6:
                threads.append(thread)
                thread = []

                if len(threads) >= data_size:
                    break
            else:
                for i, sent in enumerate(line[3:-1]):
                    words = []
                    for w in sent.split():
                        w = w.lower()
                        if test is False:
                            vocab.add(w)
                        words.append(w)
                    line[3 + i] = words

                ##################
                # Label          #
                # -1: Not sample #
                # 0-: Sample     #
                ##################
                line[-1] = -1 if line[-1][0] == '-' else int(line[-1][0])
                thread.append(line)

    return threads, vocab


def convert_word_into_string(threads):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :param vocab_word: Vocab()
    :return: threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    """

    if threads is None:
        return None

    count = 0
    for thread in threads:
        for sent in thread:
            sent[3] = ' '.join(sent[3])
            if sent[2] != '-':
                for i, r in enumerate(sent[4:-1]):
                    sent[4 + i] = ' '.join(r)
                count += 1

    print('\n\tQuestions: {:>8}'.format(count))
    return threads


class Sample(object):

    def __init__(self, context, spk_id, adr_id, responses, label, n_agents_in_ctx, max_n_agents, test=False):

        # str
        self.spk_id = spk_id
        self.adr_id = adr_id

        # 1D: n_prev_sents, 2D: max_n_words
        self.context = [c[-1] for c in context]
        # 1D: n_cands, 2D: max_n_words
        self.response = responses

        self.agent_index_dict = indexing(spk_id, context)
        # 1D: n_prev_sents, 2D: max_n_agents; one-hot vector
        self.spk_agent_one_hot_vec = get_spk_agent_one_hot_vec(context, self.agent_index_dict, max_n_agents)
        self.spk_agents = [s.index(1) for s in self.spk_agent_one_hot_vec]

        self.adr_agent_one_hot_vec = get_adr_agent_one_hot_vec(context, self.agent_index_dict, max_n_agents)
        self.adr_agents = [s.index(1) for s in self.adr_agent_one_hot_vec]

        ###################
        # Response labels #
        ###################
        false_res_label = get_false_res_label(responses, label)
        self.true_res = get_true_res_label(label, false_res_label, test)
        self.response = self.response if test else get_responses(label, false_res_label, self.response)
        self.all_response = self.response if test else get_responses(label, false_res_label, self.response)

        ####################
        # Addressee labels #
        ####################
        self.true_adr = get_adr_label(adr_id, self.agent_index_dict)
        self.adr_label_vec = get_adr_label_vec(adr_id, self.agent_index_dict, max_n_agents)

        self.n_agents_in_lctx = len(set([c[1] for c in context] + [spk_id]))
        self.binned_n_agents_in_ctx = bin_n_agents_in_ctx(n_agents_in_ctx)
        self.n_agents_in_ctx = n_agents_in_ctx


def get_responses(true_label, false_label, response):
    if true_label < false_label:
        return [response[true_label]] + response[false_label]
    return response[false_label] + [response[true_label]]


def get_adr_label(addressee_id, agent_index_dict):
    """
    :param addressee_id: the addressee of the response; int
    :param agent_index_dict: {agent id: agent index}
    """

    n_agents_lctx = len(agent_index_dict)

    # the case of including addressee in the limited context
    if addressee_id in agent_index_dict and n_agents_lctx > 1:
        true_addressee = agent_index_dict[addressee_id] - 1
    else:
        true_addressee = -1

    return true_addressee


def get_adr_label_vec(adr_id, agent_index_dict, max_n_agents):
    """
    :param adr_id: the addressee of the response; int
    :param agent_index_dict: {agent id: agent index}
    """

    y = []
    n_agents_lctx = len(agent_index_dict)

    # the case of including addressee in the limited context
    if adr_id in agent_index_dict and n_agents_lctx > 1:
        # True addressee index
        y.append(agent_index_dict[adr_id])

        # False addressee index
        for i in range(len(agent_index_dict)):
            if i not in y:
                y.append(i)

    pad = [-1 for i in range(max_n_agents-len(y))]
    y = y + pad
    return y


def get_true_res_label(true_label, false_label, test):
    if test:
        return true_label
    if true_label < false_label:
        return 0
    return 1


def get_false_res_label(response, label):
    """
    :param response: [response1, response2, ... ]
    :param label: true response label; int
    :return: int
    """
    n_responses = len(response)
    cand_indices = [r for r in range(n_responses)]
    cand_indices.remove(label)
    np.random.shuffle(cand_indices)
    return cand_indices[0]

def get_spk_agent_one_hot_vec(context, agent_index_dict, max_n_agents):
    """
    :param context: 1D: n_prev_sents, 2D: n_words
    :param agent_index_dict: {agent id: agent index}
    :param max_n_agents: the max num of agents that appear in the context (=n_prev_sents+1); int
    :return: 1D: n_prev_sents, 2D: max_n_agents
    """
    speaking_agent_one_hot_vector = []
    for c in context:
        vec = [0 for i in range(max_n_agents)]
        speaker_id = c[1]
        vec[agent_index_dict[speaker_id]] = 1
        speaking_agent_one_hot_vector.append(vec)
    return speaking_agent_one_hot_vector

def get_adr_agent_one_hot_vec(context, agent_index_dict, max_n_agents):
    """
    :param context: 1D: n_prev_sents, 2D: n_words
    :param agent_index_dict: {agent id: agent index}
    :param max_n_agents: the max num of agents that appear in the context (=n_prev_sents+1); int
    :return: 1D: n_prev_sents, 2D: max_n_agents
    """
    speaking_agent_one_hot_vector = []
    for c in context:
        vec = [0 for i in range(max_n_agents+2)]
        speaker_id = c[2]
        if speaker_id == "-":
            vec[-1] = 1
        else:
            if speaker_id in agent_index_dict:
                vec[agent_index_dict[speaker_id]] = 1
            else:
                vec[-2] = 1
        speaking_agent_one_hot_vector.append(vec)
    return speaking_agent_one_hot_vector


def indexing(responding_agent_id, context):
    agent_ids = {responding_agent_id: 0}
    for c in reversed(context):
        agent_id = c[1]
        if not agent_id in agent_ids.keys():
            agent_ids[agent_id] = len(agent_ids)
    return agent_ids


def bin_n_agents_in_ctx(n):
    if n < 6:
        return 0
    elif n < 11:
        return 1
    elif n < 16:
        return 2
    elif n < 21:
        return 3
    elif n < 31:
        return 4
    elif n < 101:
        return 5
    return 6


def statistics(samples, max_n_agents):
    show_adr_chance_level(samples)
    show_adr_upper_bound(samples, max_n_agents)
    show_n_samples_binned_ctx(samples)


def show_adr_chance_level(samples):
    total = float(len(samples))
    total_agents = 0.
    stats = defaultdict(int)

    for sample in samples:
        stats[sample.n_agents_in_ctx] += 1

    for n_agents, n_samples in stats.items():
        assert n_agents > 0
        total_agents += n_agents * n_samples

    print('\n\t  SAMPLES: {:>8}'.format(int(total)))
    print('\n\t  ADDRESSEE DETECTION CHANCE LEVEL: {:>7.2%}'.format(total / total_agents))


def show_adr_upper_bound(samples, max_n_agents):
    true_adr_stats = defaultdict(int)
    non_adr_stats = defaultdict(int)

    # sample.n_agents_in_lctx = agents appearing in the limited context (including the speaker of the response)
    for sample in samples:
        if sample.true_adr > -1:
            true_adr_stats[sample.n_agents_in_lctx] += 1
        else:
            non_adr_stats[sample.n_agents_in_lctx] += 1

    print('\n\t  ADDRESSEE DETECTION UPPER BOUND:')
    for n_agents in range(max_n_agents):
        n_agents += 1
        if n_agents in true_adr_stats:
            ttl1 = true_adr_stats[n_agents]
        else:
            ttl1 = 0
        if n_agents in non_adr_stats:
            ttl2 = non_adr_stats[n_agents]
        else:
            ttl2 = 0
        total = float(ttl1 + ttl2)

        if total == 0:
            ub = 0.
        else:
            ub = ttl1 / total

        print('\n\t\t# Cands {:>2}: {:>7.2%} | Total: {:>8} | Including true-adr: {:>8} | Not including: {:>8}'.format(
            n_agents, ub, int(total), ttl1, ttl2))
    print('\n')


def show_n_samples_binned_ctx(samples):
    ctx_stats = defaultdict(int)
    for sample in samples:
        ctx_stats[sample.binned_n_agents_in_ctx] += 1

    print('\n\t  THE BINNED NUMBER OF AGENTS IN CONTEXT:')
    for n_agents, ttl in sorted(ctx_stats.items(), key=lambda x: x[0]):
        print('\n\t\tBin {:>2}: {:>8}'.format(n_agents, ttl))
    print('\n')


def get_samples(threads, n_prev_sents, max_n_words=1000, pad=True, test=False):
    """
    :param threads: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    :param n_prev_sents: how many previous sentences are used
    :param max_n_words: how many words in a utterance are used
    :param pad: whether do the zero padding or not
    :param test: whether the dev/test set or not
    :return: samples: 1D: n_samples; elem=Sample()
    """

    if threads is None:
        return None

    print('\n\tTHREADS: {:>5}'.format(len(threads)))

    samples = []
    max_n_agents = n_prev_sents + 1

    for thread in threads:
        samples += get_one_thread_samples(thread, max_n_agents, n_prev_sents, test)

    statistics(samples, max_n_agents)

    return samples

def get_original_sent(responses, label):
    if label > -1:
        return responses[label]
    return responses[0]

def get_one_thread_samples(thread, max_n_agents, n_prev_sents, test=False):
    samples = []
    sents = []
    agents_in_ctx = set([])

    for i, sent in enumerate(thread):
        time = sent[0]
        spk_id = sent[1]
        adr_id = sent[2]
        label = sent[-1]

        context = get_context(i, sents, n_prev_sents, label, test)
        responses = sent[3:-1]
        original_sent = get_original_sent(responses, label)
        sents.append((time, spk_id, adr_id, original_sent))

        agents_in_ctx.add(spk_id)

        ################################
        # Judge if it is sample or not #
        ################################
        if is_sample(context, spk_id, adr_id, agents_in_ctx):
            sample = Sample(context=context, spk_id=spk_id, adr_id=adr_id, responses=responses, label=label,
                            n_agents_in_ctx=len(agents_in_ctx), max_n_agents=max_n_agents, test=test)
            if test:
                samples.append(sample)
            else:
                # The num of the agents in the training samples is n_agents > 1
                # -1 means that the addressee does not appear in the limited context
                if sample.true_adr > -1:
                    samples.append(sample)

    return samples

def is_sample(context, spk_id, adr_id, agents_in_ctx):
    if context is None:
        return False
    if spk_id == adr_id:
        return False
    if adr_id not in agents_in_ctx:
        return False
    return True


def get_context(i, sents, n_prev_sents, label, test=False):
    # context: 1D: n_prev_sent, 2D: (time, speaker_id, addressee_id, tokens, label)
    context = None
    if label > -1:
        if len(sents) >= n_prev_sents:
            context = sents[i - n_prev_sents:i]
        elif test:
            context = sents[:i]
    return context

def create_samples(argv, train_dataset, dev_dataset, test_dataset):
    ###########################
    # Task setting parameters #
    ###########################
    n_prev_sents = 15
    cands = train_dataset[0][0][3:-1]
    n_cands = len(cands)

    print('\n\nTASK  SETTING')
    print('\n\tResponse Candidates:%d  Contexts:%d \n' % (n_cands, n_prev_sents))

    ##########################
    # Convert words into ids #
    ##########################
    print('\n\nConverting words into ids...')
    # samples: 1D: n_threads, 2D: n_sents, 3D: (time, speaker_id, addressee_id, response, ..., label)
    train_samples = convert_word_into_string(train_dataset)
    dev_samples = convert_word_into_string(dev_dataset)
    test_samples = convert_word_into_string(test_dataset)

    print('\n\nCreating samples...')
    # samples: 1D: n_samples; 2D: Sample
    train_samples = get_samples(threads=train_samples, n_prev_sents=n_prev_sents, test=True)
    dev_samples = get_samples(threads=dev_samples, n_prev_sents=n_prev_sents, test=True)
    test_samples = get_samples(threads=test_samples, n_prev_sents=n_prev_sents, test=True)

    return train_samples, dev_samples, test_samples