import os

def PrettyPrintMatrix(attr, labelarray = None):
    os.system("color")

    listofwrongvalues = []
    for row_index in range(len(attr)):
        for element_index in range(len(attr[row_index])):
            if row_index != element_index and attr[row_index][element_index]!=0:
                listofwrongvalues.append(attr[row_index][element_index])

    listofwrongvalues.sort()
    threshold = round(listofwrongvalues[-1]/2)

    if labelarray == None:
        matrix = [["Table"]]
        for i in range(len(attr)):
            matrix[0].append("Label "+str(i+1))
        for i in range(len(attr)):
            matrix.append(["Label "+str(i+1)]+attr[i])
    else: 
        matrix = [[""]+labelarray]
        for i in range(len(attr)):
            matrix.append([labelarray[i]]+attr[i])

    s = [[str(e) for e in row] for row in matrix]
    tablelens = [[len(e) for e in row] for row in s]
    lens = [max(map(len, col)) for col in zip(*s)]

    s = []
    for row_index in range(len(matrix)):
        row = []
        for element_index in range(len(matrix[row_index])):
            int_element = matrix[row_index][element_index]
            string_element = str(int_element)
            if row_index!=0 and element_index!=0:
                if row_index == element_index:
                    string_element = "\u001b[32;1m" + string_element + "\u001b[0m"
                elif int_element<=threshold and int_element!=0:
                    string_element = "\u001b[33;1m" + string_element + "\u001b[0m"
                elif int_element>threshold and int_element!=0:
                    string_element = "\u001b[31;1m" + string_element + "\u001b[0m"
                else:
                    string_element = "\u001b[37;1m" + string_element + "\u001b[0m"
            else:
                string_element = "\u001b[1;1m" + string_element + "\u001b[0m"
            row.append(string_element)
        s.append(row)

    table = [[s[i][j]+str(" " * (lens[j] - tablelens[i][j])) for j in range(len(s[i]))] for i in range(len(s))]
    table = [" | ".join(i) for i in table]

    return '\n'.join(table)+"\nReal Label (vertical) v/s Predicted Label (horizontal)"