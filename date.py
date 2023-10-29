# this multiple-layer neural network without learning process an
# with console printing elements.
# this network copies the thinking of a person who selects a romantic partner 
# based on three factors: whether the potential candidate has an apartment, 
# his love of rock music and how attractive the candidate is.


import numpy as np
house = None
rock = None
attr = None


gender = input("Are you finding for man or woman?")

if gender.lower() == 'man':
    gender_2 = 'he'
elif gender.lower() == 'woman':
    gender_2 = 'she'

house_q = input(f"Does {gender_2} have a flat? ")

if house_q.lower() == "yes":
    house = 1
else: 
    house = 0

rock_q = input(f"Does {gender_2} like rock music? ")
if rock_q.lower() == "yes":
    rock = 1
else: 
    rock = 0



attr_q = input(f" *Hmm... is {gender_2} attractive?* ")
if attr_q.lower() == "yes":
    attr = 1
else: 
    attr = 0