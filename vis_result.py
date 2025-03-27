import re

def get_answers(log_file, split_on='-'*50, answer_note = 'SUCCESS: Local Search Response:'):
    
    with open(log_file, 'r') as file:
        output_content = file.read()

    # Split the responses based on the separator
    responses = output_content.strip().split(split_on)
    responses = responses[:-1]  # Remove the last empty response

    # Regular expression to find the part after answer_note
    pattern = fr'{re.escape(answer_note)}(.*)'
    success_pattern = re.compile(pattern, re.DOTALL)

    # find the head, relation tail in 'Evaluate this triple: {head}, {relation}, {tail}. Is this triple valid or not?', remove the str after
    triple_pattern = re.compile(r'Evaluate this triple: (.+), (.+), (.+). Is this triple valid or not\?')

    # Regular expression to find the content inside \boxed{}
    boxed_pattern = re.compile(r'\\boxed\{([^}]+)\}')
    box_pattern = re.compile(r'\\box\{([^}]+)\}')

    answers = []
    num_no_answer = 0
    triples = []
    # Process each response
    for i, response in enumerate(responses):
        
        # Find the part after answer_note
        success_match = success_pattern.search(response)
        if success_match:
            success_text = success_match.group(1).strip()
            #print(f"SUCCESS Text: {success_text}")
            # get the head, relation, tail
            triple_match = triple_pattern.search(response)
            if not triple_match:
                print(f"Processing Response {i}:")
                print(f"No triple found in this response: {response}")
                triples.append(('Error', 'Error', 'Error'))
                continue
            head = triple_match.group(1)
            relation = triple_match.group(2)
            tail = triple_match.group(3)
            triples.append((head, relation, tail))
            # Check if there's an answer in \boxed{}
            boxed_match = boxed_pattern.search(success_text)
            if boxed_match:
                answer = boxed_match.group(1)
                answers.append(answer)
                #print(f"Answer found in \\boxed{{}}: {answer}")
            else:
                box_match = box_pattern.search(success_text)
                if box_match:
                    answer = box_match.group(1)
                    answers.append(answer)
                else:
                    num_no_answer += 1
                    answers.append('E')
                    #print(f"Processing Response {i}:")
                    #print("No answer found in \\boxed{}.")
                    #print(success_text)
        else:
            #answers.append('N')
            #num_no_answer += 1
            print(f"Processing Response {i}:")
            print("No answer found in this response.")
    
    return triples, answers, num_no_answer

total_num = 0
total_num_true = 0
total_num_false = 0
total_num_e = 0
total_num_none = 0

all_answers = []
all_triples = []
for i in range(0, 9):
    log_file = f"/scratch/gpfs/jx0800/outputs/triple_validation_result_llama8B_{i}.log"
    triples, answers, num_no_answer = get_answers(log_file, split_on='-'*50, answer_note = 'Response: ')
    print(len(answers))
    total_num += len(answers)
    all_answers.extend(answers)
    all_triples.extend(triples)

    # answers is a list of 'True', 'False', 'E', or None (if no answer found)
    # compute the number of 'True' and 'False' answers
    num_true = answers.count('True') + answers.count('true')
    num_false = answers.count('False') + answers.count('false')
    num_e = answers.count('E')
    num_none = answers.count('N')

    total_num_true += num_true
    total_num_false += num_false
    total_num_e += num_e
    total_num_none += num_none

    print(f"Number of 'True' answers: {num_true}")
    print(f"Number of 'False' answers: {num_false}")  
    print(f"Number of 'E' answers: {num_e}")  
    print(f"Number of 'None' answers: {num_none}")

print(f"Total number of responses: {total_num}")
print(f"Total number of 'True' answers: {total_num_true}")
print(f"Total number of 'False' answers: {total_num_false}")    
print(f"Total number of 'E' answers: {total_num_e}")
print(f"Total number of 'None' answers: {total_num_none}")
print()
print(f"Total True rate: {total_num_true/total_num}")
print(f"Total False rate: {total_num_false/total_num}")
print(f"Total fail rate: {1-(total_num_true+total_num_false)/total_num}")

# save all_answers and all_triples to a file
import pickle
with open('/scratch/gpfs/jx0800/outputs/all_answers_triples.pkl', 'wb') as f:
    pickle.dump((all_answers, all_triples), f) 

# load all_answers and all_triples from a file
with open('/scratch/gpfs/jx0800/outputs/all_answers_triples.pkl', 'rb') as f:
    all_answers, all_triples = pickle.load(f)

# for each record, keep the triple when the answer is 'True', save it to a csv file
import pandas as pd
data = {'root': [], 'relation': [], 'tail': []}
for answer, triple in zip(all_answers, all_triples):
    if answer == 'True':
        data['root'].append(triple[0])
        data['relation'].append(triple[1])
        data['tail'].append(triple[2])

df = pd.DataFrame(data)
df.to_csv('/scratch/gpfs/jx0800/outputs/expanded_true_triples.csv', index=False)