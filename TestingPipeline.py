from Declaration.Declaration import perform_test,BERTClass,read_pretrained_model_tokenizer
import Declaration.Declaration
from Declaration import Config

def getPrediction(my_text):
    model,tokenizer = read_pretrained_model_tokenizer()
    Config.model = model
    Config.tokenizera =tokenizer

    # my_text = """I am Nabarun Barua doing class with Arjun Kumbakkara in Lucknow.
    #              Institute iNeuron is teaching us Artificial Neural Network.
    #              Everyone should Post at least one post in Facebook and Instagram in favor of iNeuron.
    #              Today is Saturday therefore in evening we will be having Bacardi White Rum in Drinks and in snacks we will eat Mutton Kabab."""

    output = perform_test(my_text,Config.model,Config.tokenizera)

    # for i in output:
    #     print(i)
    return output