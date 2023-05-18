import os
from scripts.recommender import User


def get_user_input():

    tmp = input('\nAge -> ')
    if tmp == 'x':
        exit(0)
    try:
        age = int(tmp)
        if age <= 0:
            print('Please enter valid age')
            return 1
    except Exception as e:
        print(e)
        return 1

    print('\n\nGender : ')
    print('1 - Male')
    print('2 - Female')
    tmp = input('\nGender -> ')
    if tmp == 'x':
        exit(0)
    if tmp == '1':
        gender = 'M'
    elif tmp == '2':
        gender = 'F'
    else:
        print('Please enter valid gender')
        return 1
    
    print('\n\nOccupation')
    occ = ['administrator', 'artist', 'doctor', 'educator', 'engineer', 'entertainment',
           'executive', 'healthcare', 'homemaker', 'lawyer', 'librarian', 'marketing', 
           'none', 'other', 'programmer', 'retired', 'salesman', 'scientist', 'student', 
           'technician', 'writer']
    for i in range(len(occ)):
        print(f'{i + 1} - {occ[i]}')
    tmp = input('\nOccupation -> ')
    if tmp == 'x':
        exit(0)
    try:
        tmp = int(tmp)
        print(tmp)
        try:
            occupation = occ[tmp - 1]
        except:
            print('Please enter valid occupation')
            return 1
    except Exception:
        print('Please enter valid occupation')
        return 1

    return User(age, gender, occupation)

if __name__ == '__main__':

    while True:
        os.system('cls')
        print('Movie Recommender System')
        print('Press x in any input to exit')

        user = get_user_input()

        if type(user) == User:
            movies = user.recommend()
            os.system('cls')
            print('Recommendations:')
            for movie in movies:
                print(movie)
        
        user_input = input('\nx to exit >> ')
        if user_input == 'x':
            break