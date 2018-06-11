import re

"""
Load all cleaned text files from wikipedia and convert them to clean 
line by line sentences text.
"""

# based histogram of an output, 50 time step for LSTM would be good!


#input location of text files
input_file = 'data/raw.en/englishText_'
# output location
output_file = 'data/wiki_sentences'

# refere to first file in input location
start_file_index = 0
# step size in the name of text file
step = 10000
# Token to replace all integers inside the text
dig_token = "$DGT$"
# pattern to find the digit numbers
dig_pattern = r'\b[0-9]*\.*[0-9]+\b'

# loop over all text 154 files
for i in range(0, 154):
    # hold text to write to new file
    data = ""

    # concatenate file name with step size
    input_filename = input_file + str(start_file_index) + '_' + str(start_file_index + step)

    # output filename
    output_filename = output_file + str(i) + '.txt'

    # increase start_file_index for next loop
    start_file_index += step


    print("reading: " + input_filename)

    with open(input_filename, mode="r", encoding="latin-1") as f:
        # lines here are paragraphs in Wikipedia
        for line in f:
            line = " ".join(line.split())
            words = line.split(' ')
            # if this paragraph is shorter than 10 words, go to next paragraph
            if len(words) < 10:
                continue
            # remove <doc id... produced by WikiExtractor
            if words[0] == '<doc':
                continue

            # divide paragraph to sentences
            pattern = '(\w\w..\.\ )'
            indexes = re.finditer(pattern, line)
            line = list(line)
            for j in indexes:
                m = j.span()
                line[m[1]-1] = '\n'
            line = ''.join(line)
            # replace digits with $DGT$
            line = re.sub(dig_pattern, dig_token, line)
            data += line + '\n'

    out_file = open(output_filename, mode="w")
    out_file.write(data)
    out_file.close()
