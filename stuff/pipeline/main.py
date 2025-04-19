from verify_student import verify_student

def main():
    
    print("🔍 Student Identity Verification System")
    student_id = input("Enter Student ID: ").strip()

    # Ask for test image name (optional)
    test_image_name = input("Enter test image name (default: frame_01.png): ").strip()
    if not test_image_name:
        test_image_name = "frame_01.png"

    result = verify_student(student_id, test_image_name)
    print(result)

if __name__ == "__main__":
    main()
