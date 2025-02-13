from scipy.ndimage import uniform_filter, uniform_filter1d

# filtering
arr = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
#print(uniform_filter1d(arr, axis=0, size=5, mode='wrap'))
arr1d = uniform_filter1d(arr, axis=1, size=3, mode='wrap')
print(arr1d)
arr1d1d = uniform_filter1d(arr1d, axis=0, size=3, mode='wrap')
print(arr1d1d)
print(uniform_filter(arr, size=3, mode='wrap'))
######

# filtering 3D
filter_size = 3
print("3D filtering")
print(np.reshape([1,2,3,4,5,6,7,8], (2,2,2)))
arr3D = np.array([
    [[ 1.0, 2.0 ], 
     [ 3.0, 4.0 ]],
    [[ 5.0, 6.0 ], 
     [ 7.0, 8.0 ]],
])

print(arr3D[0,0,1])
print(arr3D[0,1,0])
print(arr3D[1,0,0])

arr3D_filt2 = uniform_filter1d(arr3D, axis=2, size=filter_size, mode='wrap')
print(f"after filtering axis 2: \n {arr3D_filt2}")
arr3D_filt1 = uniform_filter1d(arr3D_filt2, axis=1, size=filter_size, mode='wrap')
print(f"after filtering axis 1: \n {arr3D_filt1}")
arr3D_filt0 = uniform_filter1d(arr3D_filt1, axis=0, size=filter_size, mode='wrap')
print(f"after filtering axis 0: \n {arr3D_filt0}")
arr3D_filt = uniform_filter(arr3D, size=filter_size, mode='wrap')
print(f"after filtering all together: \n {arr3D_filt}")
####