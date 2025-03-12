import string
import time

# Change the time at which the 'drooping letter effect' takes place here
dropping_time_delay = 0.05

alphabets = list(string.ascii_lowercase)


word = input('Please Enter a word to apply the dropping letter effect: ')


word_length = len(word)
full_print = ''

for i in range(word_length):

    x = 0
    
    if word[i] not in alphabets:
        full_print += word[i]
        print(full_print)
    
    else:
        while True:
            
            last_letter = alphabets[x]
            full_print += last_letter
            print(full_print)

            time.sleep(dropping_time_delay)

            if alphabets[x] == word[i]:
                break
            
            full_print = full_print[:-1]

            x+=1
            