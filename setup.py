from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="deploy_model",
        sources=[r"D:\DaiHoc\Intern\ThienPhuocCompany\FishNet_ThienPhuocCompany\module\deployModel.pyx"],
        # Nếu cần, bạn có thể cấu hình thêm include_dirs, library_dirs, libraries
        # nhưng với cv2 thường đã được cài đặt qua pip nên không cần thiết.
    )
]

setup(
    name="deploy_model",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
