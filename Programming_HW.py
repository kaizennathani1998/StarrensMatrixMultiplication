import os.path
import time
import argparse


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Matrix Multiplication Program")
    parser.add_argument("input_file", help="Path to the input file containing matrices")
    # Optionally, define an output file argument if needed
    # parser.add_argument("output_file", help="Path to the output file")
    return parser.parse_args()

def file_path_validation(file_path):
    """
    :function: To validate if the input file path exists
    :param file_path: file path string
    :input: File path string
    :output: Status of file path existence
    :exception: If the givenfile path does not exist
    :rtype: Exception or print
    """

    if not os.path.exists(file_path):
        raise FileExistsError(f"-> Error: The file '{file_path}' does not exist. Try Again or press 0 to end")
    elif os.path.getsize(file_path) == 0:
        raise FileExistsError(f"-> Error: File is Empty. Please try again or press 0 to End")
    print(f"File Found. Processing File...")


def parse_matrix(filename):
    # Initialize an empty list to store matrices
    matrices = []
    #line number variable
    line_no = 0
    # Initialize an empty list to store rows of the current matrix
    matrix = []
    # Initialize a variable to store the order of the current matrix
    order = 0
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            #incriment line number
            line_no = line_no+1
            # Remove leading and trailing whitespaces from the line
            line = line.strip()
            # If the line is blank, reset the matrix and order for the next matrix
            if line == "":
                buffer_matrix =[]
                matrix = []
                order = 0
                continue
            # If the order of the current matrix is not determined yet,
            # extract the order from the line
            if order == 0 and line != "":
                order = int(line)
                continue
            # Split the line by whitespace and convert each element to an integer
            row = [int(x) for x in line.split()]
            #If the line has number of integers equal to specified order of matrix append
            if len(row) == order:
                # Append the row to the current matrix
                matrix.append(row)
            # If the line does not have number of integers equal to order break and raise error
            elif len(row) != order:
                print("-> Error: The File is not in the correct format. Encountered error at line %d in the file"%line_no)
                #Initialize matrices =-1 for indicating bad file format
                matrices =-1

            # If the number of rows in the current matrix equals the order,
            # it means the current matrix is complete. Append it to the list of matrices
            # and reset the matrix for the next one
            if len(matrix) == order and matrices!=-1:
                matrices.append(matrix)
                matrix = []
    # Close the file
    file.close()
    # Return the list of matrices
    return matrices


def read_input(file_path):
    """
    :function: To parse input matrices of different orders from the user given input file
    :param: No parameter
    :input: user entered file path
    :exception: if file path does not exist and if input file is empty or ill-formatted
    :output: List of different input matrices
    :rtype: list
    """
    print("The Entered Input file path is: ",file_path)
    print("Finding File...")
    # Ask the user for input file
    #file_path = input("Please enter the file path containing input matrices:\n")

    # Check if user wants to end the program with 0 key
    if file_path == "0":
        print("Exiting code...Ciao!")
        exit()


    # Check if the file path exists and if the file is valid
    try:
        file_path_validation(file_path)
    except FileExistsError as e:
        print(e)
        # Ask the user to re-enter the file path
        file_path = input("\nPlease input a valid File Path or enter 0 to exit code:\n")
        return read_input(file_path)
    # Read the file and parse matrices
    matrices = parse_matrix(file_path)
    #If the file has bad format, it will return -1
    if matrices ==-1:
        #Prompt user for new file path
        file_path = input("\nPlease input a valid File Path or enter 0 to exit code:")
        #Re-call the read_input after taking new file name
        matrices = read_input(file_path)

    return matrices


def divide_matrix(m):
    """
    :function: Takes an input matrix of power 2 and divides it into sub-matrices of size n/2
    :input: Matrix of power 2 order
    :exception: if the input matrix is not of power to it will produce error
    :param m: parent matrix of order n
    :return: list of sub-matrices of order n/2
    """
    #Initialize empty matrix to store sub-matrices
    divided_matrices = []
    #Compute the order of input matrix
    n = len(m)
    # Check if the matrix order is not a power of 2
    if n % 2 != 0:
        raise ValueError("Matrix Order is not a power of 2")
    #Compute the n/2 for the order of sub-array
    block_size = n // 2
    #Initialize empty sub-matrix to store the submatrix in each iteration
    sub_matrix = []
    # Iterate over columns with a step size of block_size
    for i in range(0, n, block_size):
        # Iterate over rows with a step size of block_size
        for j in range(0, n, block_size):
            # Extract sub-matrix
            sub_matrix = [row[i:i + block_size] for row in m[j:j + block_size]]
            divided_matrices.append(sub_matrix)
    return divided_matrices


def matrix_arithmetic(matrix1, matrix2, op):
    """
    Performs matrix addition or subtraction.

    :param matrix1: The first matrix.
    :type matrix1: list of lists
    :param matrix2: The second matrix.
    :type matrix2: list of lists
    :param op: Type of arithmetic operation (+ or -)
    :type op: String
    :return: The result of the matrix arithmetic.
    :rtype: list of lists
    :raises ValueError: If the matrices have different dimensions or op is not '+' or '-'.
    """
    # Input validation
    if op not in ("+", "-"):
        raise ValueError("Invalid operation. Supported operations are '+' and '-'.")

    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise ValueError("Matrices must have the same dimensions for addition or subtraction.")

    result = []
    for i in range(len(matrix1)):  # Iterate over rows
        row = []
        for j in range(len(matrix1[0])):  # Iterate over columns
            if op == "+":
                row.append(matrix1[i][j] + matrix2[i][j])  # Add corresponding elements
            elif op == "-":
                row.append(matrix1[i][j] - matrix2[i][j])  # Subtract corresponding elements
        result.append(row)

    return result



def matrix_multiplication(matrix1, matrix2):
    """
    Perform matrix multiplication between two matrices.

    :param matrix1: First matrix
    :param matrix2: Second matrix
    :return: Resultant matrix, multiplication count
    """
    # Check if the matrices are compatible for multiplication
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Matrices are not compatible for multiplication")

    # Initialize the resultant matrix with zeros
    result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    # Perform matrix multiplication and count the number of multiplications
    multiplications = 0
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
                multiplications += 1

    return result,multiplications


def compute_s(mat1, mat2):
    """
    Computes the S matrices for the Strassen algorithm.

    :param mat1: List of matrices representing D matrices for the first matrix
    :param mat2: List of matrices representing D matrices for the second matrix
    :return: Tuple containing S matrices
    """
    # Compute S1 = B12 - B22
    s1 = matrix_arithmetic(mat2[2], mat2[3], "-")

    # Compute S2 = A11 + A12
    s2 = matrix_arithmetic(mat1[0], mat1[2], "+")

    # Compute S3 = A21 + A22
    s3 = matrix_arithmetic(mat1[1], mat1[3], "+")

    # Compute S4 = B21 - B11
    s4 = matrix_arithmetic(mat2[1], mat2[0], "-")

    # Compute S5 = A11 + A22
    s5 = matrix_arithmetic(mat1[0], mat1[3], "+")

    # Compute S6 = B11 + B22
    s6 = matrix_arithmetic(mat2[0], mat2[3], "+")

    # Compute S7 = A12 - A22
    s7 = matrix_arithmetic(mat1[2], mat1[3], "-")

    # Compute S8 = B21 + B22
    s8 = matrix_arithmetic(mat2[1], mat2[3], "+")

    # Compute S9 = A11 - A21
    s9 = matrix_arithmetic(mat1[0], mat1[1], "-")

    # Compute S10 = B11 + B12
    s10 = matrix_arithmetic(mat2[0], mat2[2], "+")

    return s1, s2, s3, s4, s5, s6, s7, s8, s9, s10


def compute_p(mat1, mat2, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10):
    """
    Computes the P matrices for the Strassen algorithm.

    :param mat1: List of matrices representing D matrices for the first matrix
    :param mat2: List of matrices representing D matrices for the second matrix
    :param s1-s10: S1-S10 matrix
    :return: Tuple containing P matrices
    """
    # Count number of multiplication
    multiplication = 0

    # Compute P1 = A11 * S1
    p1, m1 = matrix_multiplication(mat1[0], s1)

    # Compute P2 = S2 * B22
    p2, m2 = matrix_multiplication(s2, mat2[3])

    # Compute P3 = S3 * B11
    p3, m3 = matrix_multiplication(s3, mat2[0])

    # Compute P4 = A22 * S4
    p4, m4 = matrix_multiplication(mat1[3], s4)

    # Compute P5 = S5 * S6
    p5, m5 = matrix_multiplication(s5, s6)

    # Compute P6 = S7 * S8
    p6, m6 = matrix_multiplication(s7, s8)

    # Compute P7 = S9 * S10
    p7, m7 = matrix_multiplication(s9, s10)

    # Count total multiplications
    multiplication = m1 + m2 + m3 + m4 + m5 + m6 + m7

    return p1, p2, p3, p4, p5, p6, p7, multiplication


def compute_c(p1, p2, p3, p4, p5, p6, p7):
    """
    Computes the C matrices for the Strassen algorithm.
    :param p1-p7: P1-P7 matrix
    :return: Tuple containing C matrices
    """
    # Compute C11 = ((P5 + P4) - P2) + P6
    c11 = matrix_arithmetic(matrix_arithmetic(matrix_arithmetic(p5, p4, "+"), p2, "-"), p6, "+")

    # Compute C12 = P1 + P2
    c12 = matrix_arithmetic(p1, p2, "+")

    # Compute C21 = P3 + P4
    c21 = matrix_arithmetic(p3, p4, "+")

    # Compute C22 = ((P5 + P1) - P3) - P7
    c22 = matrix_arithmetic(matrix_arithmetic(matrix_arithmetic(p5, p1, "+"), p3, "-"), p7, "-")

    return c11, c12, c21, c22

def form_result_matrix(c11, c12, c21, c22):
    """
    Forms the matrix multiplication result from the C matrices.

    :param c11: C11 matrix
    :param c12: C12 matrix
    :param c21: C21 matrix
    :param c22: C22 matrix
    :return: Resultant matrix
    """
    # Get the order of the matrices
    n = len(c11)

    # Initialize the resultant matrix with zeros
    result = [[0 for _ in range(n * 2)] for _ in range(n * 2)]

    # Copy the C matrices into the resultant matrix
    for i in range(n):
        for j in range(n):
            result[i][j] = c11[i][j]
            result[i][j + n] = c12[i][j]
            result[i + n][j] = c21[i][j]
            result[i + n][j + n] = c22[i][j]

    return result

def starrens_multiplication(a, b):
    """
    Performs matrix multiplication using Strassen's algorithm by calling divide_matrix,
    compute_s, compute_p and compute_c functions

    :param a: First matrix
    :param b: Second matrix
    :return: returns reuslting matrix from starrens multiplication
    """
    # Step 1: Divide the matrices
    print("Order of matrices to multiply (n) = %d " % len(a))
    print("\n")
    print("Step-1: Divide the matrices in n/2 order. Here, n= %d and n/2 = %d" % (len(a), int(len(a) / 2)))
    m1 = divide_matrix(a) #Dividing Matrix-A by calling divide_matrix() function
    m2 = divide_matrix(b) #Dividing Matrix-B by calling divide_matrix() function
    print("-> Result of Matrix Division:")
    print(m1)
    print(m2)
    print("\n")

    # Step 2: Compute S matrices
    print("Step-2: Compute S matrices")
    #Computing S matrices by calling compute_s() function
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = compute_s(m1, m2)
    print("-> Result of S matrices")
    print("-> s1 =", s1)
    print("-> s2 =", s2)
    print("-> s3 =", s3)
    print("-> s4 =", s4)
    print("-> s5 =", s5)
    print("-> s6 =", s6)
    print("-> s7 =", s7)
    print("-> s8 =", s8)
    print("-> s9 =", s9)
    print("-> s10 =", s10)
    print("\n")

    # Step 3: Compute P matrices
    print("Step-3: Compute P matrices")
    #Computing P matrices by calling compute_p() function
    p1, p2, p3, p4, p5, p6, p7,multi = compute_p(m1, m2, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
    print("-> Result of P matrices")
    print("-> p1 =", p1)
    print("-> p2 =", p2)
    print("-> p3 =", p3)
    print("-> p4 =", p4)
    print("-> p5 =", p5)
    print("-> p6 =", p6)
    print("-> p7 =", p7)
    print("\n")

    # Step 4: Compute C matrices
    print("Step-4: Compute C matrices")
    #Computing C matrices by calling compute_c() function
    c11, c12, c21, c22 = compute_c(p1, p2, p3, p4, p5, p6, p7)
    print("-> Result of C matrices")
    print("-> c11 =", c11)
    print("-> c12 =", c12)
    print("-> c21 =", c21)
    print("-> c22 =", c22)
    print("\n")

    # Form the resultant matrix
    print(">>>>>>>>>>>RESULTS<<<<<<<<<<<\n")
    print("Result of Starren's Matrix Multiplication:",)
    #Displaying the Starren's matrix multiplication result in standard matrix form using form_result_matrix() function
    rss = form_result_matrix(c11, c12, c21, c22)
    for j in range(0,len(rss)):
        print(rss[j])
    print("Number of multiplications = ", multi)
    #Return multiplication output
    return rss


#Main Executing Driver
def main():
    args = parse_arguments()
    file_path = args.input_file
    # Read input matrices from the user
    print("EN.605.620.81: Algorithms for Bioinformatics\nProgrammer Name: Kaizen Nathani \t Professor Name: Prof. Sidney Rubey\n")
    print("Welcome To Matrix Multiplication Program\n")
    print("Reading the Command Line given Input File")
    matrices = read_input(file_path)
    print("Total Matrices in Input file = ",len(matrices))


    #Variable to store running time, order

    # Counter for matrix sets
    mc = 0
    print("***************************MATRIX MULTIPLICATION BEGINS***************************")
    # Iterate over matrix sets
    for i in range(0, len(matrices), 2):
        mc += 1
        # Select two matrices for multiplication
        a = matrices[i]
        b = matrices[i + 1]

        #Display the matrix set number
        print("\nMatrix Set:", mc)
        print("Input Matrices:")
        print("-> Matrice A:")
        for k in range(0,len(a)):
            print(a[k])
        print("-> Matrice B:")
        for k in range(0,len(b)):
            print(b[k])

        # Measure time taken for Strassen's algorithm multiplication
        st_time1 = time.time()
        #Ececute Starrens multiplication
        rs = starrens_multiplication(a, b)
        st_time2 = time.time()
        #Calculate running time
        st_runtime = st_time2 - st_time1

        # Print Strassen's algorithm multiplication result and runtime
        print("Running Time for Strassen's Multiplication = ", st_runtime)
        print("\n")

        # Compare with normal matrix multiplication
        print("Result of Normal Matrix Multiplication:")

        # Measure time taken for normal matrix multiplication
        start_time = time.time()
        #Call Normal matrix multiplication
        rs, multi2 = matrix_multiplication(a, b)
        end_time = time.time()
        run_time = end_time - start_time

        # Print normal matrix multiplication result and runtime
        for k in range(0, len(rs)):
            print(rs[k])
        print("Number of multiplications for Normal Multiplication = ", multi2)
        print("Running time for Normal Multiplication = ", run_time)
        # Print separator for clarity
        print("--------------------------------------------------------------------------------------------------")
        print("\n")

#If the file is called by include it would not execute main()
if __name__ == "__main__":
    main()

#/Users/kaizennathani/Downloads/LabStrassenInput2.txt