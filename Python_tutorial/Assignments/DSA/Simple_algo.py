
##his is for binart search
def test_location(cards, query, mid):
    ##test location is is tackle the case of [1,4,5,5,5,8,78] means repeating numbers
    mid_number = cards[mid]
    print("mid:", mid, ", mid_number:", mid_number)
    if mid_number == query:
        if mid-1 >= 0 and cards[mid-1] == query:
            return 'left'
        else:
            return 'found'
    elif mid_number < query:
        return 'left'
    else:
        return 'right'

def locate_card(cards, query):
    lo, hi = 0, len(cards) - 1
    
    while lo <= hi:
        print("lo:", lo, ", hi:", hi)
        mid = (lo + hi) // 2
        result = test_location(cards, query, mid)
        
        if result == 'found':
            return mid
        elif result == 'left':
            hi = mid - 1
        elif result == 'right':
            lo = mid + 1
    return -1

    '''## The Method - Revisited

Here's a systematic strategy we've applied for solving the problem:

1. State the problem clearly. Identify the input & output formats.
2. Come up with some example inputs & outputs. Try to cover all edge cases.
3. Come up with a correct solution for the problem. State it in plain English.
4. Implement the solution and test it using example inputs. Fix bugs, if any.
5. Analyze the algorithm's complexity and identify inefficiencies, if any.
6. Apply the right technique to overcome the inefficiency. Repeat steps 3 to 6.

Use this template for solving problems using this method:
 https://jovian.ai/aakashns/python-problem-solving-template

This seemingly obvious strategy will help you solve almost any programming 
problem you will face in an interview or coding assessment. 

The objective of this course is to rewire your brain to think using this method, 
by applying it over and over to different types of problems. 
This is a course about thinking about problems systematically and turning those thoughts into code.'''