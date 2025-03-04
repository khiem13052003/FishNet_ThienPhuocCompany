from module import deployModel

if __name__ == "__main__":
    folder_path = r"D:\DaiHoc\Intern\ThienPhuocCompany\data_fishNet\luoimoi_data"  # Thay đổi đường dẫn phù hợp
    processor = deployModel.ImageProcessor()
    processor.processImgFolder(folder_path)
    # processor.realTime()  
