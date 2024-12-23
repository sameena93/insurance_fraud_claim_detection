from setuptools import find_packages, setup 
#find_packages search for __init__.py file and consider as the package
# stup responsible for provide all the information about the project
from typing import List


def get_requirements()->List[str]:
    """
    function will return list of requirements"""
    req_list:List[str]=[]
    try:
        with open("requirements.txt", "r") as file:
            # lines= file.read()
            lines = file.readlines()
            for line in lines:
                req = line.strip()
                if req and req != '-e .':
                    req_list.append(req)
    except FileNotFoundError:
        # logging.info("requirement.txt fiel not found!")
        print("requirement.txt fiel not found!")

    return req_list
# print(get_requirements())

setup(
    name= 'Healthcare_insurance_fraud_claim',
    version= '0.0.1',
    author='Sameena',
    author_email= 'sameena93@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements()

)