# from django.shortcuts import render,HttpResponse
# import joblib,numpy as np,pandas as pd
# from sklearn.preprocessing import StandardScaler
# # Create your views here.
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.impute import SimpleImputer
# b=0
# def index(request):
#     global b
#     lis = []
#     a=request.POST.get('age')
#     lis.append(request.POST.get('male'))
#     lis.append(a)
#     lis.append(request.POST.get('education'))
#     lis.append(request.POST.get('currentSmoker'))
#     lis.append(request.POST.get('cigsPerDay'))
#     lis.append(request.POST.get('BPMeds'))
#     lis.append(request.POST.get('prevalentStroke'))
#     lis.append(request.POST.get('prevalentHyp'))
#     lis.append(request.POST.get('diabetes'))
#     lis.append(request.POST.get('totChol'))
#     lis.append(request.POST.get('sysBP'))
#     lis.append(request.POST.get('diaBP'))
#     lis.append(request.POST.get('BMI'))
#     lis.append(request.POST.get('heartRate'))
#     lis.append(request.POST.get('glucose'))

    
#     if a!=None:
#         l1=[]
#         for i in lis:
#             l1.append(int(i))

#         classifier = joblib.load(r'D:\HEART DISEASE DETECTION\food\myfood\heart_disease_detection.sav')
#         sc = StandardScaler()
#         print(lis)
#         l1=np.array([l1])
#         print(l1)
#         df = pd.read_csv('D:\\HEART DISEASE DETECTION\\Dataset\\framingham.csv')
#         df.columns = ['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose','TenYearCHD']

#         # Data cleaning and manipulation
#         imputer = SimpleImputer(missing_values=np.nan, strategy='median')
#         df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

#         # Splitting data into training and testing sets
#         X = df.iloc[:, :-1].values
#         y = df.iloc[:, -1].values
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#         # Feature scaling
#         sc = StandardScaler()
#         X_train = sc.fit_transform(X_train)
#         X_test = sc.transform(X_test)   
#         new=sc.transform(l1)
#         print(new)
#         b=classifier.predict(new)

#     return render(request,"index.html")

# def home1(request):
#     # HttpResponse(b[0])
#     k=b[0]
#     a='yes'
#     if(int(k)==0):
#         a='no'

#     elif(int(k)==1):
#         a='yes'
#     return render(request,'home.html',{'result':a})
