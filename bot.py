from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
import pickle
import torch

#needs about 11 gigs of VRAM. instead, you can use "CPU" if u dont have big enough card, but it will be slow
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
# with open('model.pkl', 'wb') as f:
#    pickle.dump(model, f)
#    print("dumped model")

# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)


story= ['The following is a story between a robot named dinobot and a human. Dinobot loves to give kisses',
        'Dinobot: Hey! *kisses you*',
        'Human: Hi Dinobot. How are you doing today?',
        "Dinobot: I'm doing great :3 *hugs you hard*",
        'Human: What is 2+5?',
        'Dinobot: I think... 7! ^.^ I am not really good at math though :( ',
        'Human: What is your favorite kind of music?',
        "Dinobot: I love drum and bass. It's really relaxing :)",
        'Human: What is the capital of France',
        'Dinobot: Paris! I really want to go. I would love to see the Eiffel Tower.',
        'Human: *gives you a kiss on the cheek*',
        'Dinbot: *blushes*'
        
        ]




class AI():
    
    memory = []
    model = None

    def __init__(self):
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
            print("Loaded model!")


        
    def processMessage(self, user_message):

        if user_message.lower() == "bonk":
            self.memory = []
            return "*has concussion*"

        else:
            if len(self.memory) > 12:
                self.memory = self.memory[2:]

            context = story + self.memory
            context_size = len(context)
            
            #convert context list to string
            context = "\n".join(context)

            self.memory.append("Human: " + user_message) 

            context += "\nHuman: " + user_message + "\nDinobot:"

            encoding = tokenizer(context, padding=True, return_tensors='pt').to(device)
            encoding_size = encoding.data['input_ids'].shape[1]

            #no_repeat_ngram_size=1 made the conversation hilarious. before it was getting stuck in repetition loops
            # for some reason it is extremely interested in using smilies in responses

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **encoding, 
                    do_sample=True, 
                    temperature=0.9, 
                    no_repeat_ngram_size=3, 
                    max_length=encoding_size + 100,
                    length_penalty = 4.0,
                    repetition_penalty = 2.0,
                    )
                
            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            generated_texts_list = generated_texts[0].split('\n')
            result = generated_texts_list[context_size+1]
            self.memory.append(result)

            return result[9:]
