# include <stdio.h> // standard input output
# include <stdbool.h> // for using bool
# include <string.h> // for strings
# include <math.h> // for math
# include <iostream>

// compile using gcc .\prog_to_refer_to_cuz_im_learning_c.c -o .\main.exe
// run using .\main.exe
int main(){

    int burgir = 32;
    float hrs = 0.14f;
    double pi = 3.14159265358979323846264;
    char asteroid = 'M'; // single quote for char '', not ""
    char bb[] = "BING BONG"; // string is array of chars... what the hell also they HAVE to be within "", not ''
    bool isDumb = true; // can also use 1 or 0 instead of true aor false

    printf("there are %d burgirs in da room\n", burgir); // Never forget to put \n 
    printf("it took %d hrs to eat da burgirs\n", hrs); // %d for float is a bad idea
    printf("it took %f hrs to eat da burgirs\n", hrs); // %f for float
    printf("it took %.3f hrs to eat da burgirs\n", hrs); // %.xf represents x decimal places
    printf("pi iz equal to: %lf\n", pi); // lf -> long float (6 places by default)
    printf("pi iz equal to: %.15lf\n", pi); // .xlf -> long float with x decimal places
    printf("xmen: asteroid %c\n", asteroid); // %c for char
    printf("%s\n", bb); // %s for string
    printf("can also represent bool like this: %d\n", isDumb);
    if (isDumb){
        printf("the creator of this program is dumb\n");
    }
    else {
        printf("nah he still dumb\n");
    }
    printf("gay\n");

    // VARIABLES
    // int -> integers ... 4 bytes
    // float -> single-precision ... 4 bytes
    // double -> double-precision ... 8 bytes
    // char -> 1 character ... 1 byte
    // char[] -> array of character (string)
    // bool -> true/false ... 1 byte, needs <stdbool.h>

    // FORMAT SPECIFIER
    // %d -> decimal (for int only)
    // %f -> float
    // %lf -> long float
    // %c -> character
    // %s -> char[] or string

    // %[x]d represents 'width' ... it adds the necessary amnt of spaces so that 5 is width
    printf("width = %5d\n", 5);
    // adjust on left using -
    printf("width = %-5d\n", 5);
    // preceed with 0s
    printf("width = %05d\n", 5);
    // it will not truncate any data tho
    printf("var = %1d\n", 234);
    // not show signs for +ve
    printf("positve = %d\n", 100);
    printf("negative = %d\n", -100);
    // show signs using %+d
    printf("positve = %+d\n", 100);
    printf("negative = %+d\n", -100);

    // precision (round off, not truncate)
    printf("%.1f\n", 19.99);
    printf("%.2f\n", 19.99);
    // adds the 0s
    printf("%.3f\n", 19.99);
    // num before the . represent minimmum width
    printf("%6.2f\n", 19.99);
    // works same as before
    printf("%-6.2f\n", 19.99);
    printf("%06.2f\n", 19.99);
    printf("%+6.2f\n", 19.99);

    int x = 2;
    int y = 3;
    printf("add: %d\n", x+y);
    printf("subtract: %d\n", x-y);
    printf("multiply: %d\n", x*y);
    printf("integer division %f\n", x/y); // does not retain decimal ka details
    // make any one of them float and ur good
    int a = 2;
    float b = 3.0f;
    printf("float division %f\n", a/b);
    float c = 2.0f;
    int d = 3;
    printf("float division %f\n", c/d);
    printf("modulus x/y %d\n", x%y);
    printf("modulus y/x %d\n", y%x);
    // ++, --, +=
    printf("x: %d\n", x);
    x++;
    printf("x: %d\n", x);
    x--;
    printf("x: %d\n", x);
    x+=100;
    printf("x: %d\n", x);
    x-=100;
    printf("x: %d\n", x);
    x*=2;
    printf("x: %d\n", x);
    x/=2;
    printf("x: %d\n", x);

    // defining, without assiging leads to undefined bs
    int undefined_num;
    printf("undefined num: %d\n", undefined_num);
    // better to keep some default value
    int defined_num = 0;
    float defined_float = 0.0f;
    char defined_char = '\0';   
    char defined_string[25] = "";

    // input
    printf("GIMME UR NUMBER PLS: ");
    scanf("%d", &defined_num);
    printf("GIMME UR FLOAT PLS: ");
    scanf("%f", &defined_float);

    // asking for char normally is messed up (after input) (buffer issue)
    printf("GIMME UR CHAR PLS: ");
    scanf("%c", &defined_char);
    printf("\n");

    //  use a spcae before %c
    printf("GIMME UR CHAR AGAIN PLS: ");
    scanf(" %c", &defined_char);

    // this is bad (it doesnt read whitespaces)
    // printf("GIMME UR STRING PLS: ");
    // scanf("%s", &defined_string);
    // printf("YOU STRING IS %s\n", defined_string);  

    // use this
    getchar(); // clears newline in buffer
    printf("GIMME UR STRING AGAIN AGAIN PLS: ");
    fgets(defined_string, sizeof(defined_string), stdin); // this also adds the \n from entering the line
    defined_string[strlen(defined_string)-1] = '\0'; // removes ast character (sets it no null)
    
    printf("YOU NUMBER IS %d\n", defined_num);
    printf("YOU FLOAT IS %f\n", defined_float);
    printf("YOU CHAR IS %c\n", defined_char);
    printf("YOU STRING IS %s\n", defined_string);

    // cout

    cout << "gay"

    // math functions
    // 

    return 0; // always end with number 0 -> good, anything else -> bad
}