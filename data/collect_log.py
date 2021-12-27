import os
import re


def not_empty(s):
    return s and s.strip()


def load_events(all_line_event_file):
    with open(all_line_event_file, 'r', encoding="utf-8") as f:
        all_line_temp = {}
        index = 0
        for event_id in f.readlines():
            all_line_temp[index] = int(event_id)
            index += 1
    return all_line_temp


def like_camel_to_tokens(camel_format):
    """
    process words like "addStoredBlock，StoredBlock，IOExceptionAndIOException，id，BLOCK，
    BLOCKException, --useDatabase, double-hummer, --reconnect-blocks, 0xADDRESS, R63-M0-L0-U19-A"
    """
    simple_format = []
    temp = ''
    flag = False

    if isinstance(camel_format, str):
        for i in range(len(camel_format)):
            if camel_format[i] == '-' or camel_format[i] == '_':
                simple_format.append(temp)
                temp = ''
                flag = False
            elif camel_format[i].isdigit():
                simple_format.append(temp)
                simple_format.append(camel_format[i])
                temp = ''
                flag = False
            elif camel_format[i].islower():
                if flag:
                    w = temp[-1]
                    temp = temp[:-1]
                    simple_format.append(temp)
                    temp = w + camel_format[i].lower()
                else:
                    temp += camel_format[i]
                flag = False
            else:
                if not flag:
                    simple_format.append(temp)
                    temp = ''
                temp += camel_format[i].lower()
                flag = True  
            if i == len(camel_format) - 1:
                simple_format.append(temp)
        simple_format = list(filter(not_empty, simple_format))
    return simple_format


def get_pure_events(process_project_module_pure_event_path, event_list_file, logger):
    events = []
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    if os.path.exists(event_list_file):
        logger.info("Read from log_events.txt")
        with open(event_list_file, 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                if line != "":
                    pure_line = re.sub(r'[^\w\d\/\_]+', ' ', line.strip())
                    pure_line_token = pure_line.split()
                    for i in range(len(pure_line_token)):
                        if pure_line_token[i][0] == '/':
                            pure_line_token[i] = ""
                        else:
                            pure_line_token[i] = ' '.join(like_camel_to_tokens(pure_line_token[i]))
                            if bool(re.search(r'\d', pure_line_token[i])):
                                pure_line_token[i] = ""
                            elif len(pure_line_token[i]) == 1:
                                pure_line_token[i] = ""
                            else:
                                pure_line_token[i] = pure_line_token[i].lower()
                    pure_line_token = list(filter(not_empty, pure_line_token))  
                    pure_line_token = list(filter(lambda x: x.lower() not in stopwords, pure_line_token))
                    pure_line_string = ' '.join(pure_line_token)
                    if len(pure_line_string) > 0:
                        events.append(pure_line_string)
                    else:
                        events.append("this_is_an_empty_event")
                else:
                    events.append("this_is_an_empty_event")
    else:
        logger.error("No event_list_file %s" % event_list_file)
        exit(-1)
    if os.path.exists(process_project_module_pure_event_path):
        os.remove(process_project_module_pure_event_path)
    with open(process_project_module_pure_event_path, 'w', encoding="utf-8") as writer:
        for event in events:
            writer.write(event + '\n')
    logger.info("Save pro-processing event %s." % process_project_module_pure_event_path)