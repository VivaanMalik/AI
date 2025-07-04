#include <iostream> // input output
#include <string>
#include <typeinfo>
#include <cmath>

using namespace std; // standard library

int main(){
    // endl or "\n" same same
    cout << "gay\n"; 
    cout << 3+3 << "\n";
    cout << "2 * 7 is " << 2*7 << endl;

    int myNum = 5;               // Integer (whole number without decimals)
    double myFloatNum = 5.99;    // Floating point number (with decimals)
    char myLetter = 'D';         // Character
    string myText = "Hello";     // String (text)
    bool myBoolean = true;       // Boolean (true or false)

    cout << myNum << myFloatNum << myLetter << myText << myBoolean << endl;

    int x = 5, y = 6, z = 50;
    cout << x + y + z << endl;

    x = y = z = 50;
    cout << x + y + z << endl;

    const float EaglesPerCheezburgir = 72.2782f;
    cout << EaglesPerCheezburgir << endl;

    string str; 
    cout << "Type a sentence: "; 
    getline(cin, str);
    cout << "Your sentence is: " << str << endl; 

    float f1 = 35e3;
    double d1 = 12E4;
    cout << f1 << endl;
    cout << d1 << endl;

    cout << myBoolean << endl;

    char a = 65, b = 66, c = 67;
    cout << a << endl;
    cout << b << endl;
    cout << c << endl;

    auto numberthatisautoint = 37388383;
    cout << typeid(numberthatisautoint).name() << endl;

    // std operators: + - * / % ++ -- 
    cout << numberthatisautoint << endl;
    numberthatisautoint++;
    cout << numberthatisautoint << endl;

    cout << (numberthatisautoint > x) << endl; // bool

    //asignment operators: += -= *= /= %= &= |= ^= >>= <<=
    // logical operator: && || !
    string strA = "Bing ";
    string strB = "Bong";
    string bb = strA + strB;
    cout << bb << endl;
    bb = strA.append(strB);
    cout << bb << endl;
    cout << bb.size() << endl;
    cout << bb.length() << endl;
    cout << bb[3] << endl;
    bb[4] = '_';
    cout << bb << endl;

    // special char \' \" \\ \n \t

    // math: https://www.w3schools.com/cpp/cpp_ref_math.asp

    if (x == y) {
        cout << "x = y" << endl;
    } else if (x>y) {
        cout << "x > y" << endl;
    } else {
        cout << "x = y" << endl;
    }
    
    string result = (x == y) ? "EQUAL" : "NO euqal";
    cout << result << endl;
    return 0;
}