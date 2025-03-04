# Challenge 1

number = int(input('ENter a +ve nuber: '))

if number > 9:
    print(f'The number {number} has more than one significant input')
elif number < 0:
    print(f'Error: Negative number not accepted!' )
else: 
    print(f'The number {number} has only one significant digits')


# Challenge 2

year = int(input('Enter the year: '))

if year % 400 == 0:
    print(f'The year {year} is a leap year')
elif year % 100 ==0:
    print(f'The year {year} is not a leap year')
elif year % 4 == 0:
    print(f'The year {year} is a leap year')
else:
    print(f'The year {year} is not a leap year')