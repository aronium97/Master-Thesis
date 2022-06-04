# NOTE : Please Refer Additional Info section of the ReadMe
# file for better understanding.


def stableMatching(n, menPreferences, womenPreferences):
    # Initially, all n men are unmarried
    unmarriedMen = list(range(n))
    # None of the men has a spouse yet, we denote this by the value None
    manSpouse = [None] * n
    # None of the women has a spouse yet, we denote this by the value None
    womanSpouse = [None] * n
    # Each man made 0 proposals, which means that
    # his next proposal will be to the woman number 0 in his list
    nextManChoice = [0] * n

    # While there exists at least one unmarried man:
    while unmarriedMen:
        # Pick an arbitrary unmarried man
        he = unmarriedMen[0]
        # Store his ranking in this variable for convenience
        hisPreferences = menPreferences[he]
        # Find a woman to propose to
        she = hisPreferences[nextManChoice[he]]
        # Store her ranking in this variable for convenience
        herPreferences = womenPreferences[she]
        # Find the present husband of the selected woman (it might be None)
        currentHusband = womanSpouse[she]

        # Now "he" proposes to "she".
        # Decide whether "she" accepts, and update the following fields
        # 1. manSpouse
        # 2. womanSpouse
        # 3. unmarriedMen
        # 4. nextManChoice
        if currentHusband == None:
            # No Husband case
            # "She" accepts any proposal
            womanSpouse[she] = he
            manSpouse[he] = she
            # "His" nextchoice is the next woman
            # in the hisPreferences list
            nextManChoice[he] = nextManChoice[he] + 1
            # Delete "him" from the
            # Unmarried list
            unmarriedMen.pop(0)
        else:
            # Husband exists
            # Check the preferences of the
            # current husband and that of the proposed man's
            currentIndex = herPreferences.index(currentHusband)
            hisIndex = herPreferences.index(he)
            # Accept the proposal if
            # "he" has higher preference in the herPreference list
            if currentIndex > hisIndex:
                # New stable match is found for "her"
                womanSpouse[she] = he
                manSpouse[he] = she
                nextManChoice[he] = nextManChoice[he] + 1
                # Pop the newly wed husband
                unmarriedMen.pop(0)
                # Now the previous husband is unmarried add
                # him to the unmarried list
                unmarriedMen.insert(0, currentHusband)
            else:
                nextManChoice[he] = nextManChoice[he] + 1

    return manSpouse

# Complexity Upper Bound : O(n^2)


print(stableMatching(2,[[1,0],[0,1]], [[0,1],[1,0]]))