Äø
àµ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
¼
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018Ì
 
$Adam/eeg_classifier_1/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/eeg_classifier_1/dense_7/bias/v

8Adam/eeg_classifier_1/dense_7/bias/v/Read/ReadVariableOpReadVariableOp$Adam/eeg_classifier_1/dense_7/bias/v*
_output_shapes
:*
dtype0
©
&Adam/eeg_classifier_1/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*7
shared_name(&Adam/eeg_classifier_1/dense_7/kernel/v
¢
:Adam/eeg_classifier_1/dense_7/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/eeg_classifier_1/dense_7/kernel/v*
_output_shapes
:	*
dtype0
¡
$Adam/eeg_classifier_1/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/eeg_classifier_1/dense_5/bias/v

8Adam/eeg_classifier_1/dense_5/bias/v/Read/ReadVariableOpReadVariableOp$Adam/eeg_classifier_1/dense_5/bias/v*
_output_shapes	
:*
dtype0
ª
&Adam/eeg_classifier_1/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¨*7
shared_name(&Adam/eeg_classifier_1/dense_5/kernel/v
£
:Adam/eeg_classifier_1/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/eeg_classifier_1/dense_5/kernel/v* 
_output_shapes
:
¨*
dtype0
¡
$Adam/eeg_classifier_1/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*5
shared_name&$Adam/eeg_classifier_1/dense_4/bias/v

8Adam/eeg_classifier_1/dense_4/bias/v/Read/ReadVariableOpReadVariableOp$Adam/eeg_classifier_1/dense_4/bias/v*
_output_shapes	
:¨*
dtype0
ª
&Adam/eeg_classifier_1/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
àK¨*7
shared_name(&Adam/eeg_classifier_1/dense_4/kernel/v
£
:Adam/eeg_classifier_1/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/eeg_classifier_1/dense_4/kernel/v* 
_output_shapes
:
àK¨*
dtype0
¢
%Adam/eeg_classifier_1/conv1d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_7/bias/v

9Adam/eeg_classifier_1/conv1d_7/bias/v/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_7/bias/v*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *8
shared_name)'Adam/eeg_classifier_1/conv1d_7/kernel/v
§
;Adam/eeg_classifier_1/conv1d_7/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_7/kernel/v*"
_output_shapes
:  *
dtype0
¢
%Adam/eeg_classifier_1/conv1d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_6/bias/v

9Adam/eeg_classifier_1/conv1d_6/bias/v/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_6/bias/v*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *8
shared_name)'Adam/eeg_classifier_1/conv1d_6/kernel/v
§
;Adam/eeg_classifier_1/conv1d_6/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_6/kernel/v*"
_output_shapes
:  *
dtype0
¼
2Adam/eeg_classifier_1/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/eeg_classifier_1/batch_normalization_3/beta/v
µ
FAdam/eeg_classifier_1/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp2Adam/eeg_classifier_1/batch_normalization_3/beta/v*
_output_shapes
: *
dtype0
¾
3Adam/eeg_classifier_1/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/eeg_classifier_1/batch_normalization_3/gamma/v
·
GAdam/eeg_classifier_1/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp3Adam/eeg_classifier_1/batch_normalization_3/gamma/v*
_output_shapes
: *
dtype0
¢
%Adam/eeg_classifier_1/conv1d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_5/bias/v

9Adam/eeg_classifier_1/conv1d_5/bias/v/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_5/bias/v*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *8
shared_name)'Adam/eeg_classifier_1/conv1d_5/kernel/v
§
;Adam/eeg_classifier_1/conv1d_5/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_5/kernel/v*"
_output_shapes
:  *
dtype0
¼
2Adam/eeg_classifier_1/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/eeg_classifier_1/batch_normalization_2/beta/v
µ
FAdam/eeg_classifier_1/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp2Adam/eeg_classifier_1/batch_normalization_2/beta/v*
_output_shapes
: *
dtype0
¾
3Adam/eeg_classifier_1/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/eeg_classifier_1/batch_normalization_2/gamma/v
·
GAdam/eeg_classifier_1/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp3Adam/eeg_classifier_1/batch_normalization_2/gamma/v*
_output_shapes
: *
dtype0
¢
%Adam/eeg_classifier_1/conv1d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_4/bias/v

9Adam/eeg_classifier_1/conv1d_4/bias/v/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_4/bias/v*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/eeg_classifier_1/conv1d_4/kernel/v
§
;Adam/eeg_classifier_1/conv1d_4/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_4/kernel/v*"
_output_shapes
: *
dtype0
 
$Adam/eeg_classifier_1/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/eeg_classifier_1/dense_7/bias/m

8Adam/eeg_classifier_1/dense_7/bias/m/Read/ReadVariableOpReadVariableOp$Adam/eeg_classifier_1/dense_7/bias/m*
_output_shapes
:*
dtype0
©
&Adam/eeg_classifier_1/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*7
shared_name(&Adam/eeg_classifier_1/dense_7/kernel/m
¢
:Adam/eeg_classifier_1/dense_7/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/eeg_classifier_1/dense_7/kernel/m*
_output_shapes
:	*
dtype0
¡
$Adam/eeg_classifier_1/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/eeg_classifier_1/dense_5/bias/m

8Adam/eeg_classifier_1/dense_5/bias/m/Read/ReadVariableOpReadVariableOp$Adam/eeg_classifier_1/dense_5/bias/m*
_output_shapes	
:*
dtype0
ª
&Adam/eeg_classifier_1/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¨*7
shared_name(&Adam/eeg_classifier_1/dense_5/kernel/m
£
:Adam/eeg_classifier_1/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/eeg_classifier_1/dense_5/kernel/m* 
_output_shapes
:
¨*
dtype0
¡
$Adam/eeg_classifier_1/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*5
shared_name&$Adam/eeg_classifier_1/dense_4/bias/m

8Adam/eeg_classifier_1/dense_4/bias/m/Read/ReadVariableOpReadVariableOp$Adam/eeg_classifier_1/dense_4/bias/m*
_output_shapes	
:¨*
dtype0
ª
&Adam/eeg_classifier_1/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
àK¨*7
shared_name(&Adam/eeg_classifier_1/dense_4/kernel/m
£
:Adam/eeg_classifier_1/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/eeg_classifier_1/dense_4/kernel/m* 
_output_shapes
:
àK¨*
dtype0
¢
%Adam/eeg_classifier_1/conv1d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_7/bias/m

9Adam/eeg_classifier_1/conv1d_7/bias/m/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_7/bias/m*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *8
shared_name)'Adam/eeg_classifier_1/conv1d_7/kernel/m
§
;Adam/eeg_classifier_1/conv1d_7/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_7/kernel/m*"
_output_shapes
:  *
dtype0
¢
%Adam/eeg_classifier_1/conv1d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_6/bias/m

9Adam/eeg_classifier_1/conv1d_6/bias/m/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_6/bias/m*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *8
shared_name)'Adam/eeg_classifier_1/conv1d_6/kernel/m
§
;Adam/eeg_classifier_1/conv1d_6/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_6/kernel/m*"
_output_shapes
:  *
dtype0
¼
2Adam/eeg_classifier_1/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/eeg_classifier_1/batch_normalization_3/beta/m
µ
FAdam/eeg_classifier_1/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp2Adam/eeg_classifier_1/batch_normalization_3/beta/m*
_output_shapes
: *
dtype0
¾
3Adam/eeg_classifier_1/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/eeg_classifier_1/batch_normalization_3/gamma/m
·
GAdam/eeg_classifier_1/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp3Adam/eeg_classifier_1/batch_normalization_3/gamma/m*
_output_shapes
: *
dtype0
¢
%Adam/eeg_classifier_1/conv1d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_5/bias/m

9Adam/eeg_classifier_1/conv1d_5/bias/m/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_5/bias/m*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *8
shared_name)'Adam/eeg_classifier_1/conv1d_5/kernel/m
§
;Adam/eeg_classifier_1/conv1d_5/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_5/kernel/m*"
_output_shapes
:  *
dtype0
¼
2Adam/eeg_classifier_1/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42Adam/eeg_classifier_1/batch_normalization_2/beta/m
µ
FAdam/eeg_classifier_1/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp2Adam/eeg_classifier_1/batch_normalization_2/beta/m*
_output_shapes
: *
dtype0
¾
3Adam/eeg_classifier_1/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/eeg_classifier_1/batch_normalization_2/gamma/m
·
GAdam/eeg_classifier_1/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp3Adam/eeg_classifier_1/batch_normalization_2/gamma/m*
_output_shapes
: *
dtype0
¢
%Adam/eeg_classifier_1/conv1d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%Adam/eeg_classifier_1/conv1d_4/bias/m

9Adam/eeg_classifier_1/conv1d_4/bias/m/Read/ReadVariableOpReadVariableOp%Adam/eeg_classifier_1/conv1d_4/bias/m*
_output_shapes
: *
dtype0
®
'Adam/eeg_classifier_1/conv1d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/eeg_classifier_1/conv1d_4/kernel/m
§
;Adam/eeg_classifier_1/conv1d_4/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/eeg_classifier_1/conv1d_4/kernel/m*"
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

eeg_classifier_1/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameeeg_classifier_1/dense_7/bias

1eeg_classifier_1/dense_7/bias/Read/ReadVariableOpReadVariableOpeeg_classifier_1/dense_7/bias*
_output_shapes
:*
dtype0

eeg_classifier_1/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*0
shared_name!eeg_classifier_1/dense_7/kernel

3eeg_classifier_1/dense_7/kernel/Read/ReadVariableOpReadVariableOpeeg_classifier_1/dense_7/kernel*
_output_shapes
:	*
dtype0

eeg_classifier_1/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameeeg_classifier_1/dense_5/bias

1eeg_classifier_1/dense_5/bias/Read/ReadVariableOpReadVariableOpeeg_classifier_1/dense_5/bias*
_output_shapes	
:*
dtype0

eeg_classifier_1/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¨*0
shared_name!eeg_classifier_1/dense_5/kernel

3eeg_classifier_1/dense_5/kernel/Read/ReadVariableOpReadVariableOpeeg_classifier_1/dense_5/kernel* 
_output_shapes
:
¨*
dtype0

eeg_classifier_1/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:¨*.
shared_nameeeg_classifier_1/dense_4/bias

1eeg_classifier_1/dense_4/bias/Read/ReadVariableOpReadVariableOpeeg_classifier_1/dense_4/bias*
_output_shapes	
:¨*
dtype0

eeg_classifier_1/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
àK¨*0
shared_name!eeg_classifier_1/dense_4/kernel

3eeg_classifier_1/dense_4/kernel/Read/ReadVariableOpReadVariableOpeeg_classifier_1/dense_4/kernel* 
_output_shapes
:
àK¨*
dtype0

eeg_classifier_1/conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name eeg_classifier_1/conv1d_7/bias

2eeg_classifier_1/conv1d_7/bias/Read/ReadVariableOpReadVariableOpeeg_classifier_1/conv1d_7/bias*
_output_shapes
: *
dtype0
 
 eeg_classifier_1/conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *1
shared_name" eeg_classifier_1/conv1d_7/kernel

4eeg_classifier_1/conv1d_7/kernel/Read/ReadVariableOpReadVariableOp eeg_classifier_1/conv1d_7/kernel*"
_output_shapes
:  *
dtype0

eeg_classifier_1/conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name eeg_classifier_1/conv1d_6/bias

2eeg_classifier_1/conv1d_6/bias/Read/ReadVariableOpReadVariableOpeeg_classifier_1/conv1d_6/bias*
_output_shapes
: *
dtype0
 
 eeg_classifier_1/conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *1
shared_name" eeg_classifier_1/conv1d_6/kernel

4eeg_classifier_1/conv1d_6/kernel/Read/ReadVariableOpReadVariableOp eeg_classifier_1/conv1d_6/kernel*"
_output_shapes
:  *
dtype0
Ä
6eeg_classifier_1/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86eeg_classifier_1/batch_normalization_3/moving_variance
½
Jeeg_classifier_1/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp6eeg_classifier_1/batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
¼
2eeg_classifier_1/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42eeg_classifier_1/batch_normalization_3/moving_mean
µ
Feeg_classifier_1/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp2eeg_classifier_1/batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
®
+eeg_classifier_1/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+eeg_classifier_1/batch_normalization_3/beta
§
?eeg_classifier_1/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp+eeg_classifier_1/batch_normalization_3/beta*
_output_shapes
: *
dtype0
°
,eeg_classifier_1/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,eeg_classifier_1/batch_normalization_3/gamma
©
@eeg_classifier_1/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp,eeg_classifier_1/batch_normalization_3/gamma*
_output_shapes
: *
dtype0

eeg_classifier_1/conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name eeg_classifier_1/conv1d_5/bias

2eeg_classifier_1/conv1d_5/bias/Read/ReadVariableOpReadVariableOpeeg_classifier_1/conv1d_5/bias*
_output_shapes
: *
dtype0
 
 eeg_classifier_1/conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *1
shared_name" eeg_classifier_1/conv1d_5/kernel

4eeg_classifier_1/conv1d_5/kernel/Read/ReadVariableOpReadVariableOp eeg_classifier_1/conv1d_5/kernel*"
_output_shapes
:  *
dtype0
Ä
6eeg_classifier_1/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *G
shared_name86eeg_classifier_1/batch_normalization_2/moving_variance
½
Jeeg_classifier_1/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp6eeg_classifier_1/batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0
¼
2eeg_classifier_1/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42eeg_classifier_1/batch_normalization_2/moving_mean
µ
Feeg_classifier_1/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp2eeg_classifier_1/batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
®
+eeg_classifier_1/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+eeg_classifier_1/batch_normalization_2/beta
§
?eeg_classifier_1/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp+eeg_classifier_1/batch_normalization_2/beta*
_output_shapes
: *
dtype0
°
,eeg_classifier_1/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,eeg_classifier_1/batch_normalization_2/gamma
©
@eeg_classifier_1/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp,eeg_classifier_1/batch_normalization_2/gamma*
_output_shapes
: *
dtype0

eeg_classifier_1/conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: */
shared_name eeg_classifier_1/conv1d_4/bias

2eeg_classifier_1/conv1d_4/bias/Read/ReadVariableOpReadVariableOpeeg_classifier_1/conv1d_4/bias*
_output_shapes
: *
dtype0
 
 eeg_classifier_1/conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" eeg_classifier_1/conv1d_4/kernel

4eeg_classifier_1/conv1d_4/kernel/Read/ReadVariableOpReadVariableOp eeg_classifier_1/conv1d_4/kernel*"
_output_shapes
: *
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ò
valueÇBÃ B»
®
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		batch_n_1
	
conv2
	batch_n_2
spatial_drop_1
	conv3
	avg_pool1
	conv4
spatial_drop_2
flat

dense1
dropout1

dense2
dropout2

dense3
dropout3
out
	optimizer

signatures*
ª
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18
.19
/20
021*

0
1
2
3
!4
"5
#6
$7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017*
* 
°
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
6trace_0
7trace_1
8trace_2
9trace_3* 
6
:trace_0
;trace_1
<trace_2
=trace_3* 
* 
È
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

kernel
bias
 D_jit_compiled_convolution_op*
Õ
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kaxis
	gamma
beta
moving_mean
 moving_variance*
È
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

!kernel
"bias
 R_jit_compiled_convolution_op*
Õ
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	#gamma
$beta
%moving_mean
&moving_variance*
¥
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`_random_generator* 
È
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

'kernel
(bias
 g_jit_compiled_convolution_op*

h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses* 
È
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

)kernel
*bias
 t_jit_compiled_convolution_op*
¥
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator* 

|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

+kernel
,bias*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

-kernel
.bias*
¬
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 

	keras_api* 
*
	keras_api
_random_generator* 
¬
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses

/kernel
0bias*
±
	¥iter
¦beta_1
§beta_2

¨decay
©learning_ratem«m¬m­m®!m¯"m°#m±$m²'m³(m´)mµ*m¶+m·,m¸-m¹.mº/m»0m¼v½v¾v¿vÀ!vÁ"vÂ#vÃ$vÄ'vÅ(vÆ)vÇ*vÈ+vÉ,vÊ-vË.vÌ/vÍ0vÎ*

ªserving_default* 
`Z
VARIABLE_VALUE eeg_classifier_1/conv1d_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEeeg_classifier_1/conv1d_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,eeg_classifier_1/batch_normalization_2/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+eeg_classifier_1/batch_normalization_2/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE2eeg_classifier_1/batch_normalization_2/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6eeg_classifier_1/batch_normalization_2/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE eeg_classifier_1/conv1d_5/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEeeg_classifier_1/conv1d_5/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,eeg_classifier_1/batch_normalization_3/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+eeg_classifier_1/batch_normalization_3/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2eeg_classifier_1/batch_normalization_3/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE6eeg_classifier_1/batch_normalization_3/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE eeg_classifier_1/conv1d_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEeeg_classifier_1/conv1d_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE eeg_classifier_1/conv1d_7/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEeeg_classifier_1/conv1d_7/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEeeg_classifier_1/dense_4/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEeeg_classifier_1/dense_4/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEeeg_classifier_1/dense_5/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEeeg_classifier_1/dense_5/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEeeg_classifier_1/dense_7/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEeeg_classifier_1/dense_7/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
 
0
 1
%2
&3*

0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16*

«0
¬1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

²trace_0* 

³trace_0* 
* 
 
0
1
2
 3*

0
1*
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

¹trace_0
ºtrace_1* 

»trace_0
¼trace_1* 
* 

!0
"1*

!0
"1*
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

Âtrace_0* 

Ãtrace_0* 
* 
 
#0
$1
%2
&3*

#0
$1*
* 

Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

Étrace_0
Êtrace_1* 

Ëtrace_0
Ìtrace_1* 
* 
* 
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses* 

Òtrace_0
Ótrace_1* 

Ôtrace_0
Õtrace_1* 
* 

'0
(1*

'0
(1*
* 

Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

Ûtrace_0* 

Ütrace_0* 
* 
* 
* 
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses* 

âtrace_0* 

ãtrace_0* 

)0
*1*

)0
*1*
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

étrace_0* 

êtrace_0* 
* 
* 
* 
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses* 

ðtrace_0
ñtrace_1* 

òtrace_0
ótrace_1* 
* 
* 
* 
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

ùtrace_0* 

útrace_0* 

+0
,1*

+0
,1*
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 

-0
.1*

-0
.1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 

/0
01*

/0
01*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses*

 trace_0* 

¡trace_0* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
¢	variables
£	keras_api

¤total

¥count*
M
¦	variables
§	keras_api

¨total

©count
ª
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 

0
 1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

¤0
¥1*

¢	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

¨0
©1*

¦	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
}
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_4/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/eeg_classifier_1/batch_normalization_2/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/eeg_classifier_1/batch_normalization_2/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_5/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_5/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/eeg_classifier_1/batch_normalization_3/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/eeg_classifier_1/batch_normalization_3/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_6/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_6/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_7/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_7/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/eeg_classifier_1/dense_4/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/eeg_classifier_1/dense_4/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/eeg_classifier_1/dense_5/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/eeg_classifier_1/dense_5/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/eeg_classifier_1/dense_7/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/eeg_classifier_1/dense_7/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_4/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/eeg_classifier_1/batch_normalization_2/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/eeg_classifier_1/batch_normalization_2/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_5/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_5/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE3Adam/eeg_classifier_1/batch_normalization_3/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE2Adam/eeg_classifier_1/batch_normalization_3/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_6/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_6/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE'Adam/eeg_classifier_1/conv1d_7/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE%Adam/eeg_classifier_1/conv1d_7/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/eeg_classifier_1/dense_4/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/eeg_classifier_1/dense_4/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/eeg_classifier_1/dense_5/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/eeg_classifier_1/dense_5/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUE&Adam/eeg_classifier_1/dense_7/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE$Adam/eeg_classifier_1/dense_7/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*!
shape:ÿÿÿÿÿÿÿÿÿ
·	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1 eeg_classifier_1/conv1d_4/kerneleeg_classifier_1/conv1d_4/bias2eeg_classifier_1/batch_normalization_2/moving_mean6eeg_classifier_1/batch_normalization_2/moving_variance+eeg_classifier_1/batch_normalization_2/beta,eeg_classifier_1/batch_normalization_2/gamma eeg_classifier_1/conv1d_5/kerneleeg_classifier_1/conv1d_5/bias2eeg_classifier_1/batch_normalization_3/moving_mean6eeg_classifier_1/batch_normalization_3/moving_variance+eeg_classifier_1/batch_normalization_3/beta,eeg_classifier_1/batch_normalization_3/gamma eeg_classifier_1/conv1d_6/kerneleeg_classifier_1/conv1d_6/bias eeg_classifier_1/conv1d_7/kerneleeg_classifier_1/conv1d_7/biaseeg_classifier_1/dense_4/kerneleeg_classifier_1/dense_4/biaseeg_classifier_1/dense_5/kerneleeg_classifier_1/dense_5/biaseeg_classifier_1/dense_7/kerneleeg_classifier_1/dense_7/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_603840
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ð 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4eeg_classifier_1/conv1d_4/kernel/Read/ReadVariableOp2eeg_classifier_1/conv1d_4/bias/Read/ReadVariableOp@eeg_classifier_1/batch_normalization_2/gamma/Read/ReadVariableOp?eeg_classifier_1/batch_normalization_2/beta/Read/ReadVariableOpFeeg_classifier_1/batch_normalization_2/moving_mean/Read/ReadVariableOpJeeg_classifier_1/batch_normalization_2/moving_variance/Read/ReadVariableOp4eeg_classifier_1/conv1d_5/kernel/Read/ReadVariableOp2eeg_classifier_1/conv1d_5/bias/Read/ReadVariableOp@eeg_classifier_1/batch_normalization_3/gamma/Read/ReadVariableOp?eeg_classifier_1/batch_normalization_3/beta/Read/ReadVariableOpFeeg_classifier_1/batch_normalization_3/moving_mean/Read/ReadVariableOpJeeg_classifier_1/batch_normalization_3/moving_variance/Read/ReadVariableOp4eeg_classifier_1/conv1d_6/kernel/Read/ReadVariableOp2eeg_classifier_1/conv1d_6/bias/Read/ReadVariableOp4eeg_classifier_1/conv1d_7/kernel/Read/ReadVariableOp2eeg_classifier_1/conv1d_7/bias/Read/ReadVariableOp3eeg_classifier_1/dense_4/kernel/Read/ReadVariableOp1eeg_classifier_1/dense_4/bias/Read/ReadVariableOp3eeg_classifier_1/dense_5/kernel/Read/ReadVariableOp1eeg_classifier_1/dense_5/bias/Read/ReadVariableOp3eeg_classifier_1/dense_7/kernel/Read/ReadVariableOp1eeg_classifier_1/dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_4/kernel/m/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_4/bias/m/Read/ReadVariableOpGAdam/eeg_classifier_1/batch_normalization_2/gamma/m/Read/ReadVariableOpFAdam/eeg_classifier_1/batch_normalization_2/beta/m/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_5/kernel/m/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_5/bias/m/Read/ReadVariableOpGAdam/eeg_classifier_1/batch_normalization_3/gamma/m/Read/ReadVariableOpFAdam/eeg_classifier_1/batch_normalization_3/beta/m/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_6/kernel/m/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_6/bias/m/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_7/kernel/m/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_7/bias/m/Read/ReadVariableOp:Adam/eeg_classifier_1/dense_4/kernel/m/Read/ReadVariableOp8Adam/eeg_classifier_1/dense_4/bias/m/Read/ReadVariableOp:Adam/eeg_classifier_1/dense_5/kernel/m/Read/ReadVariableOp8Adam/eeg_classifier_1/dense_5/bias/m/Read/ReadVariableOp:Adam/eeg_classifier_1/dense_7/kernel/m/Read/ReadVariableOp8Adam/eeg_classifier_1/dense_7/bias/m/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_4/kernel/v/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_4/bias/v/Read/ReadVariableOpGAdam/eeg_classifier_1/batch_normalization_2/gamma/v/Read/ReadVariableOpFAdam/eeg_classifier_1/batch_normalization_2/beta/v/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_5/kernel/v/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_5/bias/v/Read/ReadVariableOpGAdam/eeg_classifier_1/batch_normalization_3/gamma/v/Read/ReadVariableOpFAdam/eeg_classifier_1/batch_normalization_3/beta/v/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_6/kernel/v/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_6/bias/v/Read/ReadVariableOp;Adam/eeg_classifier_1/conv1d_7/kernel/v/Read/ReadVariableOp9Adam/eeg_classifier_1/conv1d_7/bias/v/Read/ReadVariableOp:Adam/eeg_classifier_1/dense_4/kernel/v/Read/ReadVariableOp8Adam/eeg_classifier_1/dense_4/bias/v/Read/ReadVariableOp:Adam/eeg_classifier_1/dense_5/kernel/v/Read/ReadVariableOp8Adam/eeg_classifier_1/dense_5/bias/v/Read/ReadVariableOp:Adam/eeg_classifier_1/dense_7/kernel/v/Read/ReadVariableOp8Adam/eeg_classifier_1/dense_7/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_604940

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename eeg_classifier_1/conv1d_4/kerneleeg_classifier_1/conv1d_4/bias,eeg_classifier_1/batch_normalization_2/gamma+eeg_classifier_1/batch_normalization_2/beta2eeg_classifier_1/batch_normalization_2/moving_mean6eeg_classifier_1/batch_normalization_2/moving_variance eeg_classifier_1/conv1d_5/kerneleeg_classifier_1/conv1d_5/bias,eeg_classifier_1/batch_normalization_3/gamma+eeg_classifier_1/batch_normalization_3/beta2eeg_classifier_1/batch_normalization_3/moving_mean6eeg_classifier_1/batch_normalization_3/moving_variance eeg_classifier_1/conv1d_6/kerneleeg_classifier_1/conv1d_6/bias eeg_classifier_1/conv1d_7/kerneleeg_classifier_1/conv1d_7/biaseeg_classifier_1/dense_4/kerneleeg_classifier_1/dense_4/biaseeg_classifier_1/dense_5/kerneleeg_classifier_1/dense_5/biaseeg_classifier_1/dense_7/kerneleeg_classifier_1/dense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount'Adam/eeg_classifier_1/conv1d_4/kernel/m%Adam/eeg_classifier_1/conv1d_4/bias/m3Adam/eeg_classifier_1/batch_normalization_2/gamma/m2Adam/eeg_classifier_1/batch_normalization_2/beta/m'Adam/eeg_classifier_1/conv1d_5/kernel/m%Adam/eeg_classifier_1/conv1d_5/bias/m3Adam/eeg_classifier_1/batch_normalization_3/gamma/m2Adam/eeg_classifier_1/batch_normalization_3/beta/m'Adam/eeg_classifier_1/conv1d_6/kernel/m%Adam/eeg_classifier_1/conv1d_6/bias/m'Adam/eeg_classifier_1/conv1d_7/kernel/m%Adam/eeg_classifier_1/conv1d_7/bias/m&Adam/eeg_classifier_1/dense_4/kernel/m$Adam/eeg_classifier_1/dense_4/bias/m&Adam/eeg_classifier_1/dense_5/kernel/m$Adam/eeg_classifier_1/dense_5/bias/m&Adam/eeg_classifier_1/dense_7/kernel/m$Adam/eeg_classifier_1/dense_7/bias/m'Adam/eeg_classifier_1/conv1d_4/kernel/v%Adam/eeg_classifier_1/conv1d_4/bias/v3Adam/eeg_classifier_1/batch_normalization_2/gamma/v2Adam/eeg_classifier_1/batch_normalization_2/beta/v'Adam/eeg_classifier_1/conv1d_5/kernel/v%Adam/eeg_classifier_1/conv1d_5/bias/v3Adam/eeg_classifier_1/batch_normalization_3/gamma/v2Adam/eeg_classifier_1/batch_normalization_3/beta/v'Adam/eeg_classifier_1/conv1d_6/kernel/v%Adam/eeg_classifier_1/conv1d_6/bias/v'Adam/eeg_classifier_1/conv1d_7/kernel/v%Adam/eeg_classifier_1/conv1d_7/bias/v&Adam/eeg_classifier_1/dense_4/kernel/v$Adam/eeg_classifier_1/dense_4/bias/v&Adam/eeg_classifier_1/dense_5/kernel/v$Adam/eeg_classifier_1/dense_5/bias/v&Adam/eeg_classifier_1/dense_7/kernel/v$Adam/eeg_classifier_1/dense_7/bias/v*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_605151Ï
¦
F
*__inference_dropout_4_layer_call_fn_604674

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_603258a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

D__inference_conv1d_7_layer_call_and_return_conditional_losses_603197

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ 
 
_user_specified_nameinputs
%
Ò
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604349

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Þ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ø
c
*__inference_dropout_4_layer_call_fn_604679

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_603355p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
m
4__inference_spatial_dropout1d_2_layer_call_fn_604464

inputs
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603031
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
L
¦
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603561
input_tensor%
conv1d_4_603501: 
conv1d_4_603503: *
batch_normalization_2_603506: *
batch_normalization_2_603508: *
batch_normalization_2_603510: *
batch_normalization_2_603512: %
conv1d_5_603515:  
conv1d_5_603517: *
batch_normalization_3_603520: *
batch_normalization_3_603522: *
batch_normalization_3_603524: *
batch_normalization_3_603526: %
conv1d_6_603530:  
conv1d_6_603532: %
conv1d_7_603536:  
conv1d_7_603538: "
dense_4_603543:
àK¨
dense_4_603545:	¨"
dense_5_603549:
¨
dense_5_603551:	!
dense_7_603555:	
dense_7_603557:
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢+spatial_dropout1d_2/StatefulPartitionedCall¢+spatial_dropout1d_3/StatefulPartitionedCallþ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCallinput_tensorconv1d_4_603501conv1d_4_603503*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_603111
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_2_603506batch_normalization_2_603508batch_normalization_2_603510batch_normalization_2_603512*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602902¨
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv1d_5_603515conv1d_5_603517*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_603142
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_603520batch_normalization_3_603522batch_normalization_3_603524batch_normalization_3_603526*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602984
+spatial_dropout1d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603031¦
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout1d_2/StatefulPartitionedCall:output:0conv1d_6_603530conv1d_6_603532*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_603174ù
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_603046
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv1d_7_603536conv1d_7_603538*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_603197·
+spatial_dropout1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0,^spatial_dropout1d_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603085ì
flatten_1/PartitionedCallPartitionedCall4spatial_dropout1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_603210
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_603543dense_4_603545*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_603223
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0,^spatial_dropout1d_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_603388
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_603549dense_5_603551*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_603247
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_603355
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_7_603555dense_7_603557*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_603271w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall,^spatial_dropout1d_2/StatefulPartitionedCall,^spatial_dropout1d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2Z
+spatial_dropout1d_2/StatefulPartitionedCall+spatial_dropout1d_2/StatefulPartitionedCall2Z
+spatial_dropout1d_3/StatefulPartitionedCall+spatial_dropout1d_3/StatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor

À
$__inference_signature_wrapper_603840
input_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14: 

unknown_15:
àK¨

unknown_16:	¨

unknown_17:
¨

unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_602831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ð

D__inference_conv1d_5_layer_call_and_return_conditional_losses_603142

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£

õ
C__inference_dense_7_layer_call_and_return_conditional_losses_603271

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
¼$
__inference__traced_save_604940
file_prefix?
;savev2_eeg_classifier_1_conv1d_4_kernel_read_readvariableop=
9savev2_eeg_classifier_1_conv1d_4_bias_read_readvariableopK
Gsavev2_eeg_classifier_1_batch_normalization_2_gamma_read_readvariableopJ
Fsavev2_eeg_classifier_1_batch_normalization_2_beta_read_readvariableopQ
Msavev2_eeg_classifier_1_batch_normalization_2_moving_mean_read_readvariableopU
Qsavev2_eeg_classifier_1_batch_normalization_2_moving_variance_read_readvariableop?
;savev2_eeg_classifier_1_conv1d_5_kernel_read_readvariableop=
9savev2_eeg_classifier_1_conv1d_5_bias_read_readvariableopK
Gsavev2_eeg_classifier_1_batch_normalization_3_gamma_read_readvariableopJ
Fsavev2_eeg_classifier_1_batch_normalization_3_beta_read_readvariableopQ
Msavev2_eeg_classifier_1_batch_normalization_3_moving_mean_read_readvariableopU
Qsavev2_eeg_classifier_1_batch_normalization_3_moving_variance_read_readvariableop?
;savev2_eeg_classifier_1_conv1d_6_kernel_read_readvariableop=
9savev2_eeg_classifier_1_conv1d_6_bias_read_readvariableop?
;savev2_eeg_classifier_1_conv1d_7_kernel_read_readvariableop=
9savev2_eeg_classifier_1_conv1d_7_bias_read_readvariableop>
:savev2_eeg_classifier_1_dense_4_kernel_read_readvariableop<
8savev2_eeg_classifier_1_dense_4_bias_read_readvariableop>
:savev2_eeg_classifier_1_dense_5_kernel_read_readvariableop<
8savev2_eeg_classifier_1_dense_5_bias_read_readvariableop>
:savev2_eeg_classifier_1_dense_7_kernel_read_readvariableop<
8savev2_eeg_classifier_1_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_4_kernel_m_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_4_bias_m_read_readvariableopR
Nsavev2_adam_eeg_classifier_1_batch_normalization_2_gamma_m_read_readvariableopQ
Msavev2_adam_eeg_classifier_1_batch_normalization_2_beta_m_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_5_kernel_m_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_5_bias_m_read_readvariableopR
Nsavev2_adam_eeg_classifier_1_batch_normalization_3_gamma_m_read_readvariableopQ
Msavev2_adam_eeg_classifier_1_batch_normalization_3_beta_m_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_6_kernel_m_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_6_bias_m_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_7_kernel_m_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_7_bias_m_read_readvariableopE
Asavev2_adam_eeg_classifier_1_dense_4_kernel_m_read_readvariableopC
?savev2_adam_eeg_classifier_1_dense_4_bias_m_read_readvariableopE
Asavev2_adam_eeg_classifier_1_dense_5_kernel_m_read_readvariableopC
?savev2_adam_eeg_classifier_1_dense_5_bias_m_read_readvariableopE
Asavev2_adam_eeg_classifier_1_dense_7_kernel_m_read_readvariableopC
?savev2_adam_eeg_classifier_1_dense_7_bias_m_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_4_kernel_v_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_4_bias_v_read_readvariableopR
Nsavev2_adam_eeg_classifier_1_batch_normalization_2_gamma_v_read_readvariableopQ
Msavev2_adam_eeg_classifier_1_batch_normalization_2_beta_v_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_5_kernel_v_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_5_bias_v_read_readvariableopR
Nsavev2_adam_eeg_classifier_1_batch_normalization_3_gamma_v_read_readvariableopQ
Msavev2_adam_eeg_classifier_1_batch_normalization_3_beta_v_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_6_kernel_v_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_6_bias_v_read_readvariableopF
Bsavev2_adam_eeg_classifier_1_conv1d_7_kernel_v_read_readvariableopD
@savev2_adam_eeg_classifier_1_conv1d_7_bias_v_read_readvariableopE
Asavev2_adam_eeg_classifier_1_dense_4_kernel_v_read_readvariableopC
?savev2_adam_eeg_classifier_1_dense_4_bias_v_read_readvariableopE
Asavev2_adam_eeg_classifier_1_dense_5_kernel_v_read_readvariableopC
?savev2_adam_eeg_classifier_1_dense_5_bias_v_read_readvariableopE
Asavev2_adam_eeg_classifier_1_dense_7_kernel_v_read_readvariableopC
?savev2_adam_eeg_classifier_1_dense_7_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ç
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHø
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ·#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_eeg_classifier_1_conv1d_4_kernel_read_readvariableop9savev2_eeg_classifier_1_conv1d_4_bias_read_readvariableopGsavev2_eeg_classifier_1_batch_normalization_2_gamma_read_readvariableopFsavev2_eeg_classifier_1_batch_normalization_2_beta_read_readvariableopMsavev2_eeg_classifier_1_batch_normalization_2_moving_mean_read_readvariableopQsavev2_eeg_classifier_1_batch_normalization_2_moving_variance_read_readvariableop;savev2_eeg_classifier_1_conv1d_5_kernel_read_readvariableop9savev2_eeg_classifier_1_conv1d_5_bias_read_readvariableopGsavev2_eeg_classifier_1_batch_normalization_3_gamma_read_readvariableopFsavev2_eeg_classifier_1_batch_normalization_3_beta_read_readvariableopMsavev2_eeg_classifier_1_batch_normalization_3_moving_mean_read_readvariableopQsavev2_eeg_classifier_1_batch_normalization_3_moving_variance_read_readvariableop;savev2_eeg_classifier_1_conv1d_6_kernel_read_readvariableop9savev2_eeg_classifier_1_conv1d_6_bias_read_readvariableop;savev2_eeg_classifier_1_conv1d_7_kernel_read_readvariableop9savev2_eeg_classifier_1_conv1d_7_bias_read_readvariableop:savev2_eeg_classifier_1_dense_4_kernel_read_readvariableop8savev2_eeg_classifier_1_dense_4_bias_read_readvariableop:savev2_eeg_classifier_1_dense_5_kernel_read_readvariableop8savev2_eeg_classifier_1_dense_5_bias_read_readvariableop:savev2_eeg_classifier_1_dense_7_kernel_read_readvariableop8savev2_eeg_classifier_1_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_4_kernel_m_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_4_bias_m_read_readvariableopNsavev2_adam_eeg_classifier_1_batch_normalization_2_gamma_m_read_readvariableopMsavev2_adam_eeg_classifier_1_batch_normalization_2_beta_m_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_5_kernel_m_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_5_bias_m_read_readvariableopNsavev2_adam_eeg_classifier_1_batch_normalization_3_gamma_m_read_readvariableopMsavev2_adam_eeg_classifier_1_batch_normalization_3_beta_m_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_6_kernel_m_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_6_bias_m_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_7_kernel_m_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_7_bias_m_read_readvariableopAsavev2_adam_eeg_classifier_1_dense_4_kernel_m_read_readvariableop?savev2_adam_eeg_classifier_1_dense_4_bias_m_read_readvariableopAsavev2_adam_eeg_classifier_1_dense_5_kernel_m_read_readvariableop?savev2_adam_eeg_classifier_1_dense_5_bias_m_read_readvariableopAsavev2_adam_eeg_classifier_1_dense_7_kernel_m_read_readvariableop?savev2_adam_eeg_classifier_1_dense_7_bias_m_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_4_kernel_v_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_4_bias_v_read_readvariableopNsavev2_adam_eeg_classifier_1_batch_normalization_2_gamma_v_read_readvariableopMsavev2_adam_eeg_classifier_1_batch_normalization_2_beta_v_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_5_kernel_v_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_5_bias_v_read_readvariableopNsavev2_adam_eeg_classifier_1_batch_normalization_3_gamma_v_read_readvariableopMsavev2_adam_eeg_classifier_1_batch_normalization_3_beta_v_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_6_kernel_v_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_6_bias_v_read_readvariableopBsavev2_adam_eeg_classifier_1_conv1d_7_kernel_v_read_readvariableop@savev2_adam_eeg_classifier_1_conv1d_7_bias_v_read_readvariableopAsavev2_adam_eeg_classifier_1_dense_4_kernel_v_read_readvariableop?savev2_adam_eeg_classifier_1_dense_4_bias_v_read_readvariableopAsavev2_adam_eeg_classifier_1_dense_5_kernel_v_read_readvariableop?savev2_adam_eeg_classifier_1_dense_5_bias_v_read_readvariableopAsavev2_adam_eeg_classifier_1_dense_7_kernel_v_read_readvariableop?savev2_adam_eeg_classifier_1_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0* 
_input_shapes
: : : : : : : :  : : : : : :  : :  : :
àK¨:¨:
¨::	:: : : : : : : : : : : : : :  : : : :  : :  : :
àK¨:¨:
¨::	:: : : : :  : : : :  : :  : :
àK¨:¨:
¨::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :($
"
_output_shapes
:  : 

_output_shapes
: :&"
 
_output_shapes
:
àK¨:!

_output_shapes	
:¨:&"
 
_output_shapes
:
¨:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :( $
"
_output_shapes
: : !

_output_shapes
: : "

_output_shapes
: : #

_output_shapes
: :($$
"
_output_shapes
:  : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: :(($
"
_output_shapes
:  : )

_output_shapes
: :(*$
"
_output_shapes
:  : +

_output_shapes
: :&,"
 
_output_shapes
:
àK¨:!-

_output_shapes	
:¨:&."
 
_output_shapes
:
¨:!/

_output_shapes	
::%0!

_output_shapes
:	: 1

_output_shapes
::(2$
"
_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: :(6$
"
_output_shapes
:  : 7

_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: :(:$
"
_output_shapes
:  : ;

_output_shapes
: :(<$
"
_output_shapes
:  : =

_output_shapes
: :&>"
 
_output_shapes
:
àK¨:!?

_output_shapes	
:¨:&@"
 
_output_shapes
:
¨:!A

_output_shapes	
::%B!

_output_shapes
:	: C

_output_shapes
::D

_output_shapes
: 
Ê

(__inference_dense_4_layer_call_fn_604611

inputs
unknown:
àK¨
	unknown_0:	¨
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_603223p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿàK: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK
 
_user_specified_nameinputs
Ò
Ò
1__inference_eeg_classifier_1_layer_call_fn_603938
input_tensor
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14: 

unknown_15:
àK¨

unknown_16:	¨

unknown_17:
¨

unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603561o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor
Ý
Ñ
6__inference_batch_normalization_2_layer_call_fn_604295

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602902|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º
m
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604569

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

D__inference_conv1d_6_layer_call_and_return_conditional_losses_603174

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿí : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
 
_user_specified_nameinputs
û	
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_603388

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
ì
2
"__inference__traced_restore_605151
file_prefixG
1assignvariableop_eeg_classifier_1_conv1d_4_kernel: ?
1assignvariableop_1_eeg_classifier_1_conv1d_4_bias: M
?assignvariableop_2_eeg_classifier_1_batch_normalization_2_gamma: L
>assignvariableop_3_eeg_classifier_1_batch_normalization_2_beta: S
Eassignvariableop_4_eeg_classifier_1_batch_normalization_2_moving_mean: W
Iassignvariableop_5_eeg_classifier_1_batch_normalization_2_moving_variance: I
3assignvariableop_6_eeg_classifier_1_conv1d_5_kernel:  ?
1assignvariableop_7_eeg_classifier_1_conv1d_5_bias: M
?assignvariableop_8_eeg_classifier_1_batch_normalization_3_gamma: L
>assignvariableop_9_eeg_classifier_1_batch_normalization_3_beta: T
Fassignvariableop_10_eeg_classifier_1_batch_normalization_3_moving_mean: X
Jassignvariableop_11_eeg_classifier_1_batch_normalization_3_moving_variance: J
4assignvariableop_12_eeg_classifier_1_conv1d_6_kernel:  @
2assignvariableop_13_eeg_classifier_1_conv1d_6_bias: J
4assignvariableop_14_eeg_classifier_1_conv1d_7_kernel:  @
2assignvariableop_15_eeg_classifier_1_conv1d_7_bias: G
3assignvariableop_16_eeg_classifier_1_dense_4_kernel:
àK¨@
1assignvariableop_17_eeg_classifier_1_dense_4_bias:	¨G
3assignvariableop_18_eeg_classifier_1_dense_5_kernel:
¨@
1assignvariableop_19_eeg_classifier_1_dense_5_bias:	F
3assignvariableop_20_eeg_classifier_1_dense_7_kernel:	?
1assignvariableop_21_eeg_classifier_1_dense_7_bias:'
assignvariableop_22_adam_iter:	 )
assignvariableop_23_adam_beta_1: )
assignvariableop_24_adam_beta_2: (
assignvariableop_25_adam_decay: 0
&assignvariableop_26_adam_learning_rate: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: #
assignvariableop_29_total: #
assignvariableop_30_count: Q
;assignvariableop_31_adam_eeg_classifier_1_conv1d_4_kernel_m: G
9assignvariableop_32_adam_eeg_classifier_1_conv1d_4_bias_m: U
Gassignvariableop_33_adam_eeg_classifier_1_batch_normalization_2_gamma_m: T
Fassignvariableop_34_adam_eeg_classifier_1_batch_normalization_2_beta_m: Q
;assignvariableop_35_adam_eeg_classifier_1_conv1d_5_kernel_m:  G
9assignvariableop_36_adam_eeg_classifier_1_conv1d_5_bias_m: U
Gassignvariableop_37_adam_eeg_classifier_1_batch_normalization_3_gamma_m: T
Fassignvariableop_38_adam_eeg_classifier_1_batch_normalization_3_beta_m: Q
;assignvariableop_39_adam_eeg_classifier_1_conv1d_6_kernel_m:  G
9assignvariableop_40_adam_eeg_classifier_1_conv1d_6_bias_m: Q
;assignvariableop_41_adam_eeg_classifier_1_conv1d_7_kernel_m:  G
9assignvariableop_42_adam_eeg_classifier_1_conv1d_7_bias_m: N
:assignvariableop_43_adam_eeg_classifier_1_dense_4_kernel_m:
àK¨G
8assignvariableop_44_adam_eeg_classifier_1_dense_4_bias_m:	¨N
:assignvariableop_45_adam_eeg_classifier_1_dense_5_kernel_m:
¨G
8assignvariableop_46_adam_eeg_classifier_1_dense_5_bias_m:	M
:assignvariableop_47_adam_eeg_classifier_1_dense_7_kernel_m:	F
8assignvariableop_48_adam_eeg_classifier_1_dense_7_bias_m:Q
;assignvariableop_49_adam_eeg_classifier_1_conv1d_4_kernel_v: G
9assignvariableop_50_adam_eeg_classifier_1_conv1d_4_bias_v: U
Gassignvariableop_51_adam_eeg_classifier_1_batch_normalization_2_gamma_v: T
Fassignvariableop_52_adam_eeg_classifier_1_batch_normalization_2_beta_v: Q
;assignvariableop_53_adam_eeg_classifier_1_conv1d_5_kernel_v:  G
9assignvariableop_54_adam_eeg_classifier_1_conv1d_5_bias_v: U
Gassignvariableop_55_adam_eeg_classifier_1_batch_normalization_3_gamma_v: T
Fassignvariableop_56_adam_eeg_classifier_1_batch_normalization_3_beta_v: Q
;assignvariableop_57_adam_eeg_classifier_1_conv1d_6_kernel_v:  G
9assignvariableop_58_adam_eeg_classifier_1_conv1d_6_bias_v: Q
;assignvariableop_59_adam_eeg_classifier_1_conv1d_7_kernel_v:  G
9assignvariableop_60_adam_eeg_classifier_1_conv1d_7_bias_v: N
:assignvariableop_61_adam_eeg_classifier_1_dense_4_kernel_v:
àK¨G
8assignvariableop_62_adam_eeg_classifier_1_dense_4_bias_v:	¨N
:assignvariableop_63_adam_eeg_classifier_1_dense_5_kernel_v:
¨G
8assignvariableop_64_adam_eeg_classifier_1_dense_5_bias_v:	M
:assignvariableop_65_adam_eeg_classifier_1_dense_7_kernel_v:	F
8assignvariableop_66_adam_eeg_classifier_1_dense_7_bias_v:
identity_68¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ê
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHû
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*
valueBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B õ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp1assignvariableop_eeg_classifier_1_conv1d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_1AssignVariableOp1assignvariableop_1_eeg_classifier_1_conv1d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_2AssignVariableOp?assignvariableop_2_eeg_classifier_1_batch_normalization_2_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_3AssignVariableOp>assignvariableop_3_eeg_classifier_1_batch_normalization_2_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_4AssignVariableOpEassignvariableop_4_eeg_classifier_1_batch_normalization_2_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_5AssignVariableOpIassignvariableop_5_eeg_classifier_1_batch_normalization_2_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_6AssignVariableOp3assignvariableop_6_eeg_classifier_1_conv1d_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_7AssignVariableOp1assignvariableop_7_eeg_classifier_1_conv1d_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOp_8AssignVariableOp?assignvariableop_8_eeg_classifier_1_batch_normalization_3_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_9AssignVariableOp>assignvariableop_9_eeg_classifier_1_batch_normalization_3_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_10AssignVariableOpFassignvariableop_10_eeg_classifier_1_batch_normalization_3_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_11AssignVariableOpJassignvariableop_11_eeg_classifier_1_batch_normalization_3_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_12AssignVariableOp4assignvariableop_12_eeg_classifier_1_conv1d_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_13AssignVariableOp2assignvariableop_13_eeg_classifier_1_conv1d_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_14AssignVariableOp4assignvariableop_14_eeg_classifier_1_conv1d_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_15AssignVariableOp2assignvariableop_15_eeg_classifier_1_conv1d_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_16AssignVariableOp3assignvariableop_16_eeg_classifier_1_dense_4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_17AssignVariableOp1assignvariableop_17_eeg_classifier_1_dense_4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_18AssignVariableOp3assignvariableop_18_eeg_classifier_1_dense_5_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_19AssignVariableOp1assignvariableop_19_eeg_classifier_1_dense_5_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_20AssignVariableOp3assignvariableop_20_eeg_classifier_1_dense_7_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_21AssignVariableOp1assignvariableop_21_eeg_classifier_1_dense_7_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_iterIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_adam_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_learning_rateIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_totalIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOpassignvariableop_30_countIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_31AssignVariableOp;assignvariableop_31_adam_eeg_classifier_1_conv1d_4_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_32AssignVariableOp9assignvariableop_32_adam_eeg_classifier_1_conv1d_4_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_33AssignVariableOpGassignvariableop_33_adam_eeg_classifier_1_batch_normalization_2_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_34AssignVariableOpFassignvariableop_34_adam_eeg_classifier_1_batch_normalization_2_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_35AssignVariableOp;assignvariableop_35_adam_eeg_classifier_1_conv1d_5_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_36AssignVariableOp9assignvariableop_36_adam_eeg_classifier_1_conv1d_5_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_37AssignVariableOpGassignvariableop_37_adam_eeg_classifier_1_batch_normalization_3_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_38AssignVariableOpFassignvariableop_38_adam_eeg_classifier_1_batch_normalization_3_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_39AssignVariableOp;assignvariableop_39_adam_eeg_classifier_1_conv1d_6_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_40AssignVariableOp9assignvariableop_40_adam_eeg_classifier_1_conv1d_6_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_41AssignVariableOp;assignvariableop_41_adam_eeg_classifier_1_conv1d_7_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_42AssignVariableOp9assignvariableop_42_adam_eeg_classifier_1_conv1d_7_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_43AssignVariableOp:assignvariableop_43_adam_eeg_classifier_1_dense_4_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_44AssignVariableOp8assignvariableop_44_adam_eeg_classifier_1_dense_4_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_45AssignVariableOp:assignvariableop_45_adam_eeg_classifier_1_dense_5_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_46AssignVariableOp8assignvariableop_46_adam_eeg_classifier_1_dense_5_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_47AssignVariableOp:assignvariableop_47_adam_eeg_classifier_1_dense_7_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_48AssignVariableOp8assignvariableop_48_adam_eeg_classifier_1_dense_7_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_49AssignVariableOp;assignvariableop_49_adam_eeg_classifier_1_conv1d_4_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_50AssignVariableOp9assignvariableop_50_adam_eeg_classifier_1_conv1d_4_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_51AssignVariableOpGassignvariableop_51_adam_eeg_classifier_1_batch_normalization_2_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_52AssignVariableOpFassignvariableop_52_adam_eeg_classifier_1_batch_normalization_2_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_53AssignVariableOp;assignvariableop_53_adam_eeg_classifier_1_conv1d_5_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_54AssignVariableOp9assignvariableop_54_adam_eeg_classifier_1_conv1d_5_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_55AssignVariableOpGassignvariableop_55_adam_eeg_classifier_1_batch_normalization_3_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_56AssignVariableOpFassignvariableop_56_adam_eeg_classifier_1_batch_normalization_3_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_57AssignVariableOp;assignvariableop_57_adam_eeg_classifier_1_conv1d_6_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_58AssignVariableOp9assignvariableop_58_adam_eeg_classifier_1_conv1d_6_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_59AssignVariableOp;assignvariableop_59_adam_eeg_classifier_1_conv1d_7_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_60AssignVariableOp9assignvariableop_60_adam_eeg_classifier_1_conv1d_7_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_61AssignVariableOp:assignvariableop_61_adam_eeg_classifier_1_dense_4_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_62AssignVariableOp8assignvariableop_62_adam_eeg_classifier_1_dense_4_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_63AssignVariableOp:assignvariableop_63_adam_eeg_classifier_1_dense_5_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_64AssignVariableOp8assignvariableop_64_adam_eeg_classifier_1_dense_5_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_65AssignVariableOp:assignvariableop_65_adam_eeg_classifier_1_dense_7_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_66AssignVariableOp8assignvariableop_66_adam_eeg_classifier_1_dense_7_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_68IdentityIdentity_67:output:0^NoOp_1*
T0*
_output_shapes
: þ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_68Identity_68:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ß

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602855

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û
Ú
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604244
input_tensorJ
4conv1d_4_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_4_biasadd_readvariableop_resource: K
=batch_normalization_2_assignmovingavg_readvariableop_resource: M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource: @
2batch_normalization_2_cast_readvariableop_resource: B
4batch_normalization_2_cast_1_readvariableop_resource: J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_5_biasadd_readvariableop_resource: K
=batch_normalization_3_assignmovingavg_readvariableop_resource: M
?batch_normalization_3_assignmovingavg_1_readvariableop_resource: @
2batch_normalization_3_cast_readvariableop_resource: B
4batch_normalization_3_cast_1_readvariableop_resource: J
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_6_biasadd_readvariableop_resource: J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_7_biasadd_readvariableop_resource: :
&dense_4_matmul_readvariableop_resource:
àK¨6
'dense_4_biasadd_readvariableop_resource:	¨:
&dense_5_matmul_readvariableop_resource:
¨6
'dense_5_biasadd_readvariableop_resource:	9
&dense_7_matmul_readvariableop_resource:	5
'dense_7_biasadd_readvariableop_resource:
identity¢%batch_normalization_2/AssignMovingAvg¢4batch_normalization_2/AssignMovingAvg/ReadVariableOp¢'batch_normalization_2/AssignMovingAvg_1¢6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_2/Cast/ReadVariableOp¢+batch_normalization_2/Cast_1/ReadVariableOp¢%batch_normalization_3/AssignMovingAvg¢4batch_normalization_3/AssignMovingAvg/ReadVariableOp¢'batch_normalization_3/AssignMovingAvg_1¢6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp¢)batch_normalization_3/Cast/ReadVariableOp¢+batch_normalization_3/Cast_1/ReadVariableOp¢conv1d_4/BiasAdd/ReadVariableOp¢+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_5/BiasAdd/ReadVariableOp¢+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_6/BiasAdd/ReadVariableOp¢+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_7/BiasAdd/ReadVariableOp¢+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOpi
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d_4/Conv1D/ExpandDims
ExpandDimsinput_tensor'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: È
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ä
"batch_normalization_2/moments/meanMeanconv1d_4/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
: Í
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv1d_4/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ä
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
  
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Ã
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
: º
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0É
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
: À
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ¯
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: ©
%batch_normalization_2/batchnorm/mul_1Mulconv1d_4/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: ­
#batch_normalization_2/batchnorm/subSub1batch_normalization_2/Cast/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ¹
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ·
conv1d_5/Conv1D/ExpandDims
ExpandDims)batch_normalization_2/batchnorm/add_1:z:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  É
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
paddingVALID*
strides

conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí g
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ä
"batch_normalization_3/moments/meanMeanconv1d_5/Relu:activations:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*"
_output_shapes
: Í
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferenceconv1d_5/Relu:activations:03batch_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ä
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
  
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<®
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Ã
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0*
T0*
_output_shapes
: º
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 
%batch_normalization_3/AssignMovingAvgAssignSubVariableOp=batch_normalization_3_assignmovingavg_readvariableop_resource-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<²
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0É
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0*
T0*
_output_shapes
: À
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 
'batch_normalization_3/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_3_assignmovingavg_1_readvariableop_resource/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: ¯
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: ©
%batch_normalization_3/batchnorm/mul_1Mulconv1d_5/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí ª
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: ­
#batch_normalization_3/batchnorm/subSub1batch_normalization_3/Cast/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ¹
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí r
spatial_dropout1d_2/ShapeShape)batch_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:q
'spatial_dropout1d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)spatial_dropout1d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)spatial_dropout1d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!spatial_dropout1d_2/strided_sliceStridedSlice"spatial_dropout1d_2/Shape:output:00spatial_dropout1d_2/strided_slice/stack:output:02spatial_dropout1d_2/strided_slice/stack_1:output:02spatial_dropout1d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)spatial_dropout1d_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+spatial_dropout1d_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+spatial_dropout1d_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
#spatial_dropout1d_2/strided_slice_1StridedSlice"spatial_dropout1d_2/Shape:output:02spatial_dropout1d_2/strided_slice_1/stack:output:04spatial_dropout1d_2/strided_slice_1/stack_1:output:04spatial_dropout1d_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
!spatial_dropout1d_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @´
spatial_dropout1d_2/dropout/MulMul)batch_normalization_3/batchnorm/add_1:z:0*spatial_dropout1d_2/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí t
2spatial_dropout1d_2/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ý
0spatial_dropout1d_2/dropout/random_uniform/shapePack*spatial_dropout1d_2/strided_slice:output:0;spatial_dropout1d_2/dropout/random_uniform/shape/1:output:0,spatial_dropout1d_2/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Ç
8spatial_dropout1d_2/dropout/random_uniform/RandomUniformRandomUniform9spatial_dropout1d_2/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0o
*spatial_dropout1d_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?æ
(spatial_dropout1d_2/dropout/GreaterEqualGreaterEqualAspatial_dropout1d_2/dropout/random_uniform/RandomUniform:output:03spatial_dropout1d_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 spatial_dropout1d_2/dropout/CastCast,spatial_dropout1d_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
!spatial_dropout1d_2/dropout/Mul_1Mul#spatial_dropout1d_2/dropout/Mul:z:0$spatial_dropout1d_2/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí i
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ³
conv1d_6/Conv1D/ExpandDims
ExpandDims%spatial_dropout1d_2/dropout/Mul_1:z:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí ¤
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  É
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
paddingVALID*
strides

conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_6/BiasAddBiasAdd conv1d_6/Conv1D/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè g
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè d
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :±
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè Æ
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ *
ksize
*
paddingVALID*
strides

average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ *
squeeze_dims
i
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ²
conv1d_7/Conv1D/ExpandDims
ExpandDims$average_pooling1d_1/Squeeze:output:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ ¤
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  É
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
paddingVALID*
strides

conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_7/BiasAddBiasAdd conv1d_7/Conv1D/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ g
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ d
spatial_dropout1d_3/ShapeShapeconv1d_7/Relu:activations:0*
T0*
_output_shapes
:q
'spatial_dropout1d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)spatial_dropout1d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)spatial_dropout1d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!spatial_dropout1d_3/strided_sliceStridedSlice"spatial_dropout1d_3/Shape:output:00spatial_dropout1d_3/strided_slice/stack:output:02spatial_dropout1d_3/strided_slice/stack_1:output:02spatial_dropout1d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
)spatial_dropout1d_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+spatial_dropout1d_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+spatial_dropout1d_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:½
#spatial_dropout1d_3/strided_slice_1StridedSlice"spatial_dropout1d_3/Shape:output:02spatial_dropout1d_3/strided_slice_1/stack:output:04spatial_dropout1d_3/strided_slice_1/stack_1:output:04spatial_dropout1d_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
!spatial_dropout1d_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @¦
spatial_dropout1d_3/dropout/MulMulconv1d_7/Relu:activations:0*spatial_dropout1d_3/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ t
2spatial_dropout1d_3/dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ý
0spatial_dropout1d_3/dropout/random_uniform/shapePack*spatial_dropout1d_3/strided_slice:output:0;spatial_dropout1d_3/dropout/random_uniform/shape/1:output:0,spatial_dropout1d_3/strided_slice_1:output:0*
N*
T0*
_output_shapes
:Ç
8spatial_dropout1d_3/dropout/random_uniform/RandomUniformRandomUniform9spatial_dropout1d_3/dropout/random_uniform/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
dtype0o
*spatial_dropout1d_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?æ
(spatial_dropout1d_3/dropout/GreaterEqualGreaterEqualAspatial_dropout1d_3/dropout/random_uniform/RandomUniform:output:03spatial_dropout1d_3/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 spatial_dropout1d_3/dropout/CastCast,spatial_dropout1d_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
!spatial_dropout1d_3/dropout/Mul_1Mul#spatial_dropout1d_3/dropout/Mul:z:0$spatial_dropout1d_3/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà%  
flatten_1/ReshapeReshape%spatial_dropout1d_3/dropout/Mul_1:z:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
àK¨*
dtype0
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨a
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨\
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_3/dropout/MulMuldense_4/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨a
dropout_3/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:¡
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*
dtype0e
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
¨*
dtype0
dense_5/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_4/dropout/MulMuldense_5/Relu:activations:0 dropout_4/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dropout_4/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:¡
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Å
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_7/MatMulMatMuldropout_4/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
NoOpNoOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp&^batch_normalization_3/AssignMovingAvg5^batch_normalization_3/AssignMovingAvg/ReadVariableOp(^batch_normalization_3/AssignMovingAvg_17^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2N
%batch_normalization_3/AssignMovingAvg%batch_normalization_3/AssignMovingAvg2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_3/AssignMovingAvg_1'batch_normalization_3/AssignMovingAvg_12p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor
Ý
k
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_603046

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

D__inference_conv1d_4_layer_call_and_return_conditional_losses_604269

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

)__inference_conv1d_5_layer_call_fn_604358

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_603142t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý

)__inference_conv1d_7_layer_call_fn_604538

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_603197t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´ : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ 
 
_user_specified_nameinputs
£
n
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604591

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :­
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:¨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?³
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿo
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
Í
1__inference_eeg_classifier_1_layer_call_fn_603325
input_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14: 

unknown_15:
àK¨

unknown_16:	¨

unknown_17:
¨

unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603278o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ß

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602937

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_604684

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_604637

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
®
F
*__inference_flatten_1_layer_call_fn_604596

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_603210a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¯ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ 
 
_user_specified_nameinputs
£
n
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604491

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :­
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:¨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?³
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿo
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§E


L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603278
input_tensor%
conv1d_4_603112: 
conv1d_4_603114: *
batch_normalization_2_603117: *
batch_normalization_2_603119: *
batch_normalization_2_603121: *
batch_normalization_2_603123: %
conv1d_5_603143:  
conv1d_5_603145: *
batch_normalization_3_603148: *
batch_normalization_3_603150: *
batch_normalization_3_603152: *
batch_normalization_3_603154: %
conv1d_6_603175:  
conv1d_6_603177: %
conv1d_7_603198:  
conv1d_7_603200: "
dense_4_603224:
àK¨
dense_4_603226:	¨"
dense_5_603248:
¨
dense_5_603250:	!
dense_7_603272:	
dense_7_603274:
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_7/StatefulPartitionedCallþ
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCallinput_tensorconv1d_4_603112conv1d_4_603114*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_603111
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_2_603117batch_normalization_2_603119batch_normalization_2_603121batch_normalization_2_603123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602855¨
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv1d_5_603143conv1d_5_603145*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_603142
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_603148batch_normalization_3_603150batch_normalization_3_603152batch_normalization_3_603154*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602937
#spatial_dropout1d_2/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603004
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout1d_2/PartitionedCall:output:0conv1d_6_603175conv1d_6_603177*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_603174ù
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_603046
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv1d_7_603198conv1d_7_603200*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_603197ù
#spatial_dropout1d_3/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603058ä
flatten_1/PartitionedCallPartitionedCall,spatial_dropout1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_603210
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_603224dense_4_603226*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_603223à
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_603234
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_603248dense_5_603250*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_603247à
dropout_4/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_603258
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_7_603272dense_7_603274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_603271w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor
Ê

(__inference_dense_5_layer_call_fn_604658

inputs
unknown:
¨
	unknown_0:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_603247p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
ß
Ñ
6__inference_batch_normalization_3_layer_call_fn_604387

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602937|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
%
Ò
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604454

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Þ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_603258

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
m
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603004

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_604649

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs

P
4__inference_average_pooling1d_1_layer_call_fn_604521

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_603046v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
n
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603085

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :­
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:¨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?³
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿo
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

P
4__inference_spatial_dropout1d_2_layer_call_fn_604459

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603004v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û	
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_603355

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

D__inference_conv1d_6_layer_call_and_return_conditional_losses_604516

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿí : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
 
_user_specified_nameinputs
ß

Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604315

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

P
4__inference_spatial_dropout1d_3_layer_call_fn_604559

inputs
identityÓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603058v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
Ò
1__inference_eeg_classifier_1_layer_call_fn_603889
input_tensor
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14: 

unknown_15:
àK¨

unknown_16:	¨

unknown_17:
¨

unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinput_tensorunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603278o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor
%
Ò
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602902

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Þ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£
n
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603031

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
dropout/MulMulinputsdropout/Const:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
dropout/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :­
dropout/random_uniform/shapePackstrided_slice:output:0'dropout/random_uniform/shape/1:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:¨
$dropout/random_uniform/RandomUniformRandomUniform%dropout/random_uniform/shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?³
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ|
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿo
IdentityIdentitydropout/Mul_1:z:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

æ
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604053
input_tensorJ
4conv1d_4_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_4_biasadd_readvariableop_resource: @
2batch_normalization_2_cast_readvariableop_resource: B
4batch_normalization_2_cast_1_readvariableop_resource: B
4batch_normalization_2_cast_2_readvariableop_resource: B
4batch_normalization_2_cast_3_readvariableop_resource: J
4conv1d_5_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_5_biasadd_readvariableop_resource: @
2batch_normalization_3_cast_readvariableop_resource: B
4batch_normalization_3_cast_1_readvariableop_resource: B
4batch_normalization_3_cast_2_readvariableop_resource: B
4batch_normalization_3_cast_3_readvariableop_resource: J
4conv1d_6_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_6_biasadd_readvariableop_resource: J
4conv1d_7_conv1d_expanddims_1_readvariableop_resource:  6
(conv1d_7_biasadd_readvariableop_resource: :
&dense_4_matmul_readvariableop_resource:
àK¨6
'dense_4_biasadd_readvariableop_resource:	¨:
&dense_5_matmul_readvariableop_resource:
¨6
'dense_5_biasadd_readvariableop_resource:	9
&dense_7_matmul_readvariableop_resource:	5
'dense_7_biasadd_readvariableop_resource:
identity¢)batch_normalization_2/Cast/ReadVariableOp¢+batch_normalization_2/Cast_1/ReadVariableOp¢+batch_normalization_2/Cast_2/ReadVariableOp¢+batch_normalization_2/Cast_3/ReadVariableOp¢)batch_normalization_3/Cast/ReadVariableOp¢+batch_normalization_3/Cast_1/ReadVariableOp¢+batch_normalization_3/Cast_2/ReadVariableOp¢+batch_normalization_3/Cast_3/ReadVariableOp¢conv1d_4/BiasAdd/ReadVariableOp¢+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_5/BiasAdd/ReadVariableOp¢+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_6/BiasAdd/ReadVariableOp¢+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp¢conv1d_7/BiasAdd/ReadVariableOp¢+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢dense_5/BiasAdd/ReadVariableOp¢dense_5/MatMul/ReadVariableOp¢dense_7/BiasAdd/ReadVariableOp¢dense_7/MatMul/ReadVariableOpi
conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
conv1d_4/Conv1D/ExpandDims
ExpandDimsinput_tensor'conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_4/Conv1D/ExpandDims_1
ExpandDims3conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: È
conv1d_4/Conv1DConv2D#conv1d_4/Conv1D/ExpandDims:output:0%conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

conv1d_4/Conv1D/SqueezeSqueezeconv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_4/BiasAddBiasAdd conv1d_4/Conv1D/Squeeze:output:0'conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ g
conv1d_4/ReluReluconv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
)batch_normalization_2/Cast/ReadVariableOpReadVariableOp2batch_normalization_2_cast_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_2/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_2/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_2/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
#batch_normalization_2/batchnorm/addAddV23batch_normalization_2/Cast_1/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ¯
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:03batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: ©
%batch_normalization_2/batchnorm/mul_1Mulconv1d_4/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ­
%batch_normalization_2/batchnorm/mul_2Mul1batch_normalization_2/Cast/ReadVariableOp:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: ¯
#batch_normalization_2/batchnorm/subSub3batch_normalization_2/Cast_2/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ¹
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ i
conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ·
conv1d_5/Conv1D/ExpandDims
ExpandDims)batch_normalization_2/batchnorm/add_1:z:0'conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¤
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_5/Conv1D/ExpandDims_1
ExpandDims3conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  É
conv1d_5/Conv1DConv2D#conv1d_5/Conv1D/ExpandDims:output:0%conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
paddingVALID*
strides

conv1d_5/Conv1D/SqueezeSqueezeconv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_5/BiasAdd/ReadVariableOpReadVariableOp(conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_5/BiasAddBiasAdd conv1d_5/Conv1D/Squeeze:output:0'conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí g
conv1d_5/ReluReluconv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
)batch_normalization_3/Cast/ReadVariableOpReadVariableOp2batch_normalization_3_cast_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_3/Cast_1/ReadVariableOpReadVariableOp4batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_3/Cast_2/ReadVariableOpReadVariableOp4batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0
+batch_normalization_3/Cast_3/ReadVariableOpReadVariableOp4batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:¶
#batch_normalization_3/batchnorm/addAddV23batch_normalization_3/Cast_1/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: ¯
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:03batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: ©
%batch_normalization_3/batchnorm/mul_1Mulconv1d_5/Relu:activations:0'batch_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí ­
%batch_normalization_3/batchnorm/mul_2Mul1batch_normalization_3/Cast/ReadVariableOp:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: ¯
#batch_normalization_3/batchnorm/subSub3batch_normalization_3/Cast_2/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ¹
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
spatial_dropout1d_2/IdentityIdentity)batch_normalization_3/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí i
conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ³
conv1d_6/Conv1D/ExpandDims
ExpandDims%spatial_dropout1d_2/Identity:output:0'conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí ¤
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_6/Conv1D/ExpandDims_1
ExpandDims3conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  É
conv1d_6/Conv1DConv2D#conv1d_6/Conv1D/ExpandDims:output:0%conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
paddingVALID*
strides

conv1d_6/Conv1D/SqueezeSqueezeconv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_6/BiasAdd/ReadVariableOpReadVariableOp(conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_6/BiasAddBiasAdd conv1d_6/Conv1D/Squeeze:output:0'conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè g
conv1d_6/ReluReluconv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè d
"average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :±
average_pooling1d_1/ExpandDims
ExpandDimsconv1d_6/Relu:activations:0+average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè Æ
average_pooling1d_1/AvgPoolAvgPool'average_pooling1d_1/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ *
ksize
*
paddingVALID*
strides

average_pooling1d_1/SqueezeSqueeze$average_pooling1d_1/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ *
squeeze_dims
i
conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ²
conv1d_7/Conv1D/ExpandDims
ExpandDims$average_pooling1d_1/Squeeze:output:0'conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ ¤
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0b
 conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_7/Conv1D/ExpandDims_1
ExpandDims3conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  É
conv1d_7/Conv1DConv2D#conv1d_7/Conv1D/ExpandDims:output:0%conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
paddingVALID*
strides

conv1d_7/Conv1D/SqueezeSqueezeconv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ
conv1d_7/BiasAdd/ReadVariableOpReadVariableOp(conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_7/BiasAddBiasAdd conv1d_7/Conv1D/Squeeze:output:0'conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ g
conv1d_7/ReluReluconv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ |
spatial_dropout1d_3/IdentityIdentityconv1d_7/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ `
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà%  
flatten_1/ReshapeReshape%spatial_dropout1d_3/Identity:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
àK¨*
dtype0
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨a
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨m
dropout_3/IdentityIdentitydense_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
¨*
dtype0
dense_5/MatMulMatMuldropout_3/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout_4/IdentityIdentitydense_5/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_7/MatMulMatMuldropout_4/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp*^batch_normalization_2/Cast/ReadVariableOp,^batch_normalization_2/Cast_1/ReadVariableOp,^batch_normalization_2/Cast_2/ReadVariableOp,^batch_normalization_2/Cast_3/ReadVariableOp*^batch_normalization_3/Cast/ReadVariableOp,^batch_normalization_3/Cast_1/ReadVariableOp,^batch_normalization_3/Cast_2/ReadVariableOp,^batch_normalization_3/Cast_3/ReadVariableOp ^conv1d_4/BiasAdd/ReadVariableOp,^conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_5/BiasAdd/ReadVariableOp,^conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_6/BiasAdd/ReadVariableOp,^conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_7/BiasAdd/ReadVariableOp,^conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2V
)batch_normalization_2/Cast/ReadVariableOp)batch_normalization_2/Cast/ReadVariableOp2Z
+batch_normalization_2/Cast_1/ReadVariableOp+batch_normalization_2/Cast_1/ReadVariableOp2Z
+batch_normalization_2/Cast_2/ReadVariableOp+batch_normalization_2/Cast_2/ReadVariableOp2Z
+batch_normalization_2/Cast_3/ReadVariableOp+batch_normalization_2/Cast_3/ReadVariableOp2V
)batch_normalization_3/Cast/ReadVariableOp)batch_normalization_3/Cast/ReadVariableOp2Z
+batch_normalization_3/Cast_1/ReadVariableOp+batch_normalization_3/Cast_1/ReadVariableOp2Z
+batch_normalization_3/Cast_2/ReadVariableOp+batch_normalization_3/Cast_2/ReadVariableOp2Z
+batch_normalization_3/Cast_3/ReadVariableOp+batch_normalization_3/Cast_3/ReadVariableOp2B
conv1d_4/BiasAdd/ReadVariableOpconv1d_4/BiasAdd/ReadVariableOp2Z
+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_5/BiasAdd/ReadVariableOpconv1d_5/BiasAdd/ReadVariableOp2Z
+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_6/BiasAdd/ReadVariableOpconv1d_6/BiasAdd/ReadVariableOp2Z
+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_7/BiasAdd/ReadVariableOpconv1d_7/BiasAdd/ReadVariableOp2Z
+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:Z V
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinput_tensor
Ð

D__inference_conv1d_7_layer_call_and_return_conditional_losses_604554

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ 
 
_user_specified_nameinputs
Ý
Ñ
6__inference_batch_normalization_3_layer_call_fn_604400

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602984|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Á
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_604602

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà%  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàKY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¯ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ 
 
_user_specified_nameinputs
E
ý	
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603720
input_1%
conv1d_4_603660: 
conv1d_4_603662: *
batch_normalization_2_603665: *
batch_normalization_2_603667: *
batch_normalization_2_603669: *
batch_normalization_2_603671: %
conv1d_5_603674:  
conv1d_5_603676: *
batch_normalization_3_603679: *
batch_normalization_3_603681: *
batch_normalization_3_603683: *
batch_normalization_3_603685: %
conv1d_6_603689:  
conv1d_6_603691: %
conv1d_7_603695:  
conv1d_7_603697: "
dense_4_603702:
àK¨
dense_4_603704:	¨"
dense_5_603708:
¨
dense_5_603710:	!
dense_7_603714:	
dense_7_603716:
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_7/StatefulPartitionedCallù
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_4_603660conv1d_4_603662*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_603111
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_2_603665batch_normalization_2_603667batch_normalization_2_603669batch_normalization_2_603671*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602855¨
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv1d_5_603674conv1d_5_603676*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_603142
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_603679batch_normalization_3_603681batch_normalization_3_603683batch_normalization_3_603685*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602937
#spatial_dropout1d_2/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603004
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall,spatial_dropout1d_2/PartitionedCall:output:0conv1d_6_603689conv1d_6_603691*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_603174ù
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_603046
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv1d_7_603695conv1d_7_603697*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_603197ù
#spatial_dropout1d_3/PartitionedCallPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603058ä
flatten_1/PartitionedCallPartitionedCall,spatial_dropout1d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_603210
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_603702dense_4_603704*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_603223à
dropout_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_603234
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_5_603708dense_5_603710*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_603247à
dropout_4/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_603258
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_4/PartitionedCall:output:0dense_7_603714dense_7_603716*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_603271w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ð

D__inference_conv1d_5_layer_call_and_return_conditional_losses_604374

inputsA
+conv1d_expanddims_1_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ®
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
%
Ò
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602984

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: *
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: 
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢Cast/ReadVariableOp¢Cast_1/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ¢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_1/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: k
batchnorm/subSubCast/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Þ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï

D__inference_conv1d_4_layer_call_and_return_conditional_losses_603111

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

(__inference_dense_7_layer_call_fn_604705

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_603271o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
L
¡
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603783
input_1%
conv1d_4_603723: 
conv1d_4_603725: *
batch_normalization_2_603728: *
batch_normalization_2_603730: *
batch_normalization_2_603732: *
batch_normalization_2_603734: %
conv1d_5_603737:  
conv1d_5_603739: *
batch_normalization_3_603742: *
batch_normalization_3_603744: *
batch_normalization_3_603746: *
batch_normalization_3_603748: %
conv1d_6_603752:  
conv1d_6_603754: %
conv1d_7_603758:  
conv1d_7_603760: "
dense_4_603765:
àK¨
dense_4_603767:	¨"
dense_5_603771:
¨
dense_5_603773:	!
dense_7_603777:	
dense_7_603779:
identity¢-batch_normalization_2/StatefulPartitionedCall¢-batch_normalization_3/StatefulPartitionedCall¢ conv1d_4/StatefulPartitionedCall¢ conv1d_5/StatefulPartitionedCall¢ conv1d_6/StatefulPartitionedCall¢ conv1d_7/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢+spatial_dropout1d_2/StatefulPartitionedCall¢+spatial_dropout1d_3/StatefulPartitionedCallù
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_4_603723conv1d_4_603725*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_603111
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0batch_normalization_2_603728batch_normalization_2_603730batch_normalization_2_603732batch_normalization_2_603734*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602902¨
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0conv1d_5_603737conv1d_5_603739*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_5_layer_call_and_return_conditional_losses_603142
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0batch_normalization_3_603742batch_normalization_3_603744batch_normalization_3_603746batch_normalization_3_603748*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_602984
+spatial_dropout1d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_603031¦
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall4spatial_dropout1d_2/StatefulPartitionedCall:output:0conv1d_6_603752conv1d_6_603754*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_603174ù
#average_pooling1d_1/PartitionedCallPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_603046
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling1d_1/PartitionedCall:output:0conv1d_7_603758conv1d_7_603760*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_7_layer_call_and_return_conditional_losses_603197·
+spatial_dropout1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0,^spatial_dropout1d_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603085ì
flatten_1/PartitionedCallPartitionedCall4spatial_dropout1d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_603210
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_603765dense_4_603767*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_603223
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0,^spatial_dropout1d_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_603388
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_5_603771dense_5_603773*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_603247
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_603355
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_4/StatefulPartitionedCall:output:0dense_7_603777dense_7_603779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_603271w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall,^spatial_dropout1d_2/StatefulPartitionedCall,^spatial_dropout1d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2Z
+spatial_dropout1d_2/StatefulPartitionedCall+spatial_dropout1d_2/StatefulPartitionedCall2Z
+spatial_dropout1d_3/StatefulPartitionedCall+spatial_dropout1d_3/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¦

÷
C__inference_dense_5_layer_call_and_return_conditional_losses_603247

inputs2
matmul_readvariableop_resource:
¨.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¨*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
ß

Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604420

inputs*
cast_readvariableop_resource: ,
cast_1_readvariableop_resource: ,
cast_2_readvariableop_resource: ,
cast_3_readvariableop_resource: 
identity¢Cast/ReadVariableOp¢Cast_1/ReadVariableOp¢Cast_2/ReadVariableOp¢Cast_3/ReadVariableOpl
Cast/ReadVariableOpReadVariableOpcast_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_1/ReadVariableOpReadVariableOpcast_1_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_2/ReadVariableOpReadVariableOpcast_2_readvariableop_resource*
_output_shapes
: *
dtype0p
Cast_3/ReadVariableOpReadVariableOpcast_3_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:t
batchnorm/addAddV2Cast_1/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: m
batchnorm/mulMulbatchnorm/Rsqrt:y:0Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ k
batchnorm/mul_2MulCast/ReadVariableOp:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: m
batchnorm/subSubCast_2/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¤
NoOpNoOp^Cast/ReadVariableOp^Cast_1/ReadVariableOp^Cast_2/ReadVariableOp^Cast_3/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 2*
Cast/ReadVariableOpCast/ReadVariableOp2.
Cast_1/ReadVariableOpCast_1/ReadVariableOp2.
Cast_2/ReadVariableOpCast_2/ReadVariableOp2.
Cast_3/ReadVariableOpCast_3/ReadVariableOp:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ý

)__inference_conv1d_6_layer_call_fn_604500

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_6_layer_call_and_return_conditional_losses_603174t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿí : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
 
_user_specified_nameinputs
Ý
k
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_604529

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

÷
C__inference_dense_4_layer_call_and_return_conditional_losses_604622

inputs2
matmul_readvariableop_resource:
àK¨.
biasadd_readvariableop_resource:	¨
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
àK¨*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿàK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK
 
_user_specified_nameinputs
ø
c
*__inference_dropout_3_layer_call_fn_604632

inputs
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_603388p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
Ã
Í
1__inference_eeg_classifier_1_layer_call_fn_603657
input_1
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10:  

unknown_11:  

unknown_12:  

unknown_13:  

unknown_14: 

unknown_15:
àK¨

unknown_16:	¨

unknown_17:
¨

unknown_18:	

unknown_19:	

unknown_20:
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603561o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¦
F
*__inference_dropout_3_layer_call_fn_604627

inputs
identity´
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_603234a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
û	
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_604696

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

)__inference_conv1d_4_layer_call_fn_604253

inputs
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv1d_4_layer_call_and_return_conditional_losses_603111t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
m
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603058

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
Ñ
6__inference_batch_normalization_2_layer_call_fn_604282

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_602855|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
º
m
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604469

inputs

identity_1d
IdentityIdentityinputs*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿq

Identity_1IdentityIdentity:output:0*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

÷
C__inference_dense_5_layer_call_and_return_conditional_losses_604669

inputs2
matmul_readvariableop_resource:
¨.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¨*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
£

õ
C__inference_dense_7_layer_call_and_return_conditional_losses_604716

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

÷
C__inference_dense_4_layer_call_and_return_conditional_losses_603223

inputs2
matmul_readvariableop_resource:
àK¨.
biasadd_readvariableop_resource:	¨
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
àK¨*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿàK: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK
 
_user_specified_nameinputs
á
m
4__inference_spatial_dropout1d_3_layer_call_fn_604564

inputs
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_603085
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
¢
!__inference__wrapped_model_602831
input_1[
Eeeg_classifier_1_conv1d_4_conv1d_expanddims_1_readvariableop_resource: G
9eeg_classifier_1_conv1d_4_biasadd_readvariableop_resource: Q
Ceeg_classifier_1_batch_normalization_2_cast_readvariableop_resource: S
Eeeg_classifier_1_batch_normalization_2_cast_1_readvariableop_resource: S
Eeeg_classifier_1_batch_normalization_2_cast_2_readvariableop_resource: S
Eeeg_classifier_1_batch_normalization_2_cast_3_readvariableop_resource: [
Eeeg_classifier_1_conv1d_5_conv1d_expanddims_1_readvariableop_resource:  G
9eeg_classifier_1_conv1d_5_biasadd_readvariableop_resource: Q
Ceeg_classifier_1_batch_normalization_3_cast_readvariableop_resource: S
Eeeg_classifier_1_batch_normalization_3_cast_1_readvariableop_resource: S
Eeeg_classifier_1_batch_normalization_3_cast_2_readvariableop_resource: S
Eeeg_classifier_1_batch_normalization_3_cast_3_readvariableop_resource: [
Eeeg_classifier_1_conv1d_6_conv1d_expanddims_1_readvariableop_resource:  G
9eeg_classifier_1_conv1d_6_biasadd_readvariableop_resource: [
Eeeg_classifier_1_conv1d_7_conv1d_expanddims_1_readvariableop_resource:  G
9eeg_classifier_1_conv1d_7_biasadd_readvariableop_resource: K
7eeg_classifier_1_dense_4_matmul_readvariableop_resource:
àK¨G
8eeg_classifier_1_dense_4_biasadd_readvariableop_resource:	¨K
7eeg_classifier_1_dense_5_matmul_readvariableop_resource:
¨G
8eeg_classifier_1_dense_5_biasadd_readvariableop_resource:	J
7eeg_classifier_1_dense_7_matmul_readvariableop_resource:	F
8eeg_classifier_1_dense_7_biasadd_readvariableop_resource:
identity¢:eeg_classifier_1/batch_normalization_2/Cast/ReadVariableOp¢<eeg_classifier_1/batch_normalization_2/Cast_1/ReadVariableOp¢<eeg_classifier_1/batch_normalization_2/Cast_2/ReadVariableOp¢<eeg_classifier_1/batch_normalization_2/Cast_3/ReadVariableOp¢:eeg_classifier_1/batch_normalization_3/Cast/ReadVariableOp¢<eeg_classifier_1/batch_normalization_3/Cast_1/ReadVariableOp¢<eeg_classifier_1/batch_normalization_3/Cast_2/ReadVariableOp¢<eeg_classifier_1/batch_normalization_3/Cast_3/ReadVariableOp¢0eeg_classifier_1/conv1d_4/BiasAdd/ReadVariableOp¢<eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp¢0eeg_classifier_1/conv1d_5/BiasAdd/ReadVariableOp¢<eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp¢0eeg_classifier_1/conv1d_6/BiasAdd/ReadVariableOp¢<eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp¢0eeg_classifier_1/conv1d_7/BiasAdd/ReadVariableOp¢<eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp¢/eeg_classifier_1/dense_4/BiasAdd/ReadVariableOp¢.eeg_classifier_1/dense_4/MatMul/ReadVariableOp¢/eeg_classifier_1/dense_5/BiasAdd/ReadVariableOp¢.eeg_classifier_1/dense_5/MatMul/ReadVariableOp¢/eeg_classifier_1/dense_7/BiasAdd/ReadVariableOp¢.eeg_classifier_1/dense_7/MatMul/ReadVariableOpz
/eeg_classifier_1/conv1d_4/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿ·
+eeg_classifier_1/conv1d_4/Conv1D/ExpandDims
ExpandDimsinput_18eeg_classifier_1/conv1d_4/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
<eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEeeg_classifier_1_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0s
1eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1
ExpandDimsDeeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp:value:0:eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: û
 eeg_classifier_1/conv1d_4/Conv1DConv2D4eeg_classifier_1/conv1d_4/Conv1D/ExpandDims:output:06eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
µ
(eeg_classifier_1/conv1d_4/Conv1D/SqueezeSqueeze)eeg_classifier_1/conv1d_4/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0eeg_classifier_1/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp9eeg_classifier_1_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ð
!eeg_classifier_1/conv1d_4/BiasAddBiasAdd1eeg_classifier_1/conv1d_4/Conv1D/Squeeze:output:08eeg_classifier_1/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
eeg_classifier_1/conv1d_4/ReluRelu*eeg_classifier_1/conv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ º
:eeg_classifier_1/batch_normalization_2/Cast/ReadVariableOpReadVariableOpCeeg_classifier_1_batch_normalization_2_cast_readvariableop_resource*
_output_shapes
: *
dtype0¾
<eeg_classifier_1/batch_normalization_2/Cast_1/ReadVariableOpReadVariableOpEeeg_classifier_1_batch_normalization_2_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0¾
<eeg_classifier_1/batch_normalization_2/Cast_2/ReadVariableOpReadVariableOpEeeg_classifier_1_batch_normalization_2_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0¾
<eeg_classifier_1/batch_normalization_2/Cast_3/ReadVariableOpReadVariableOpEeeg_classifier_1_batch_normalization_2_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0{
6eeg_classifier_1/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
4eeg_classifier_1/batch_normalization_2/batchnorm/addAddV2Deeg_classifier_1/batch_normalization_2/Cast_1/ReadVariableOp:value:0?eeg_classifier_1/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: 
6eeg_classifier_1/batch_normalization_2/batchnorm/RsqrtRsqrt8eeg_classifier_1/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: â
4eeg_classifier_1/batch_normalization_2/batchnorm/mulMul:eeg_classifier_1/batch_normalization_2/batchnorm/Rsqrt:y:0Deeg_classifier_1/batch_normalization_2/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: Ü
6eeg_classifier_1/batch_normalization_2/batchnorm/mul_1Mul,eeg_classifier_1/conv1d_4/Relu:activations:08eeg_classifier_1/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ à
6eeg_classifier_1/batch_normalization_2/batchnorm/mul_2MulBeeg_classifier_1/batch_normalization_2/Cast/ReadVariableOp:value:08eeg_classifier_1/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: â
4eeg_classifier_1/batch_normalization_2/batchnorm/subSubDeeg_classifier_1/batch_normalization_2/Cast_2/ReadVariableOp:value:0:eeg_classifier_1/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ì
6eeg_classifier_1/batch_normalization_2/batchnorm/add_1AddV2:eeg_classifier_1/batch_normalization_2/batchnorm/mul_1:z:08eeg_classifier_1/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
/eeg_classifier_1/conv1d_5/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿê
+eeg_classifier_1/conv1d_5/Conv1D/ExpandDims
ExpandDims:eeg_classifier_1/batch_normalization_2/batchnorm/add_1:z:08eeg_classifier_1/conv1d_5/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ Æ
<eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEeeg_classifier_1_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1
ExpandDimsDeeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp:value:0:eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ü
 eeg_classifier_1/conv1d_5/Conv1DConv2D4eeg_classifier_1/conv1d_5/Conv1D/ExpandDims:output:06eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
paddingVALID*
strides
µ
(eeg_classifier_1/conv1d_5/Conv1D/SqueezeSqueeze)eeg_classifier_1/conv1d_5/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí *
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0eeg_classifier_1/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp9eeg_classifier_1_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ð
!eeg_classifier_1/conv1d_5/BiasAddBiasAdd1eeg_classifier_1/conv1d_5/Conv1D/Squeeze:output:08eeg_classifier_1/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí 
eeg_classifier_1/conv1d_5/ReluRelu*eeg_classifier_1/conv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí º
:eeg_classifier_1/batch_normalization_3/Cast/ReadVariableOpReadVariableOpCeeg_classifier_1_batch_normalization_3_cast_readvariableop_resource*
_output_shapes
: *
dtype0¾
<eeg_classifier_1/batch_normalization_3/Cast_1/ReadVariableOpReadVariableOpEeeg_classifier_1_batch_normalization_3_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0¾
<eeg_classifier_1/batch_normalization_3/Cast_2/ReadVariableOpReadVariableOpEeeg_classifier_1_batch_normalization_3_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0¾
<eeg_classifier_1/batch_normalization_3/Cast_3/ReadVariableOpReadVariableOpEeeg_classifier_1_batch_normalization_3_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0{
6eeg_classifier_1/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:é
4eeg_classifier_1/batch_normalization_3/batchnorm/addAddV2Deeg_classifier_1/batch_normalization_3/Cast_1/ReadVariableOp:value:0?eeg_classifier_1/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 
6eeg_classifier_1/batch_normalization_3/batchnorm/RsqrtRsqrt8eeg_classifier_1/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: â
4eeg_classifier_1/batch_normalization_3/batchnorm/mulMul:eeg_classifier_1/batch_normalization_3/batchnorm/Rsqrt:y:0Deeg_classifier_1/batch_normalization_3/Cast_3/ReadVariableOp:value:0*
T0*
_output_shapes
: Ü
6eeg_classifier_1/batch_normalization_3/batchnorm/mul_1Mul,eeg_classifier_1/conv1d_5/Relu:activations:08eeg_classifier_1/batch_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí à
6eeg_classifier_1/batch_normalization_3/batchnorm/mul_2MulBeeg_classifier_1/batch_normalization_3/Cast/ReadVariableOp:value:08eeg_classifier_1/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: â
4eeg_classifier_1/batch_normalization_3/batchnorm/subSubDeeg_classifier_1/batch_normalization_3/Cast_2/ReadVariableOp:value:0:eeg_classifier_1/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: ì
6eeg_classifier_1/batch_normalization_3/batchnorm/add_1AddV2:eeg_classifier_1/batch_normalization_3/batchnorm/mul_1:z:08eeg_classifier_1/batch_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí ¬
-eeg_classifier_1/spatial_dropout1d_2/IdentityIdentity:eeg_classifier_1/batch_normalization_3/batchnorm/add_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿí z
/eeg_classifier_1/conv1d_6/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿæ
+eeg_classifier_1/conv1d_6/Conv1D/ExpandDims
ExpandDims6eeg_classifier_1/spatial_dropout1d_2/Identity:output:08eeg_classifier_1/conv1d_6/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿí Æ
<eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEeeg_classifier_1_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1
ExpandDimsDeeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp:value:0:eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ü
 eeg_classifier_1/conv1d_6/Conv1DConv2D4eeg_classifier_1/conv1d_6/Conv1D/ExpandDims:output:06eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
paddingVALID*
strides
µ
(eeg_classifier_1/conv1d_6/Conv1D/SqueezeSqueeze)eeg_classifier_1/conv1d_6/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè *
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0eeg_classifier_1/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp9eeg_classifier_1_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ð
!eeg_classifier_1/conv1d_6/BiasAddBiasAdd1eeg_classifier_1/conv1d_6/Conv1D/Squeeze:output:08eeg_classifier_1/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè 
eeg_classifier_1/conv1d_6/ReluRelu*eeg_classifier_1/conv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè u
3eeg_classifier_1/average_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ä
/eeg_classifier_1/average_pooling1d_1/ExpandDims
ExpandDims,eeg_classifier_1/conv1d_6/Relu:activations:0<eeg_classifier_1/average_pooling1d_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿè è
,eeg_classifier_1/average_pooling1d_1/AvgPoolAvgPool8eeg_classifier_1/average_pooling1d_1/ExpandDims:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ *
ksize
*
paddingVALID*
strides
¼
,eeg_classifier_1/average_pooling1d_1/SqueezeSqueeze5eeg_classifier_1/average_pooling1d_1/AvgPool:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ *
squeeze_dims
z
/eeg_classifier_1/conv1d_7/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ýÿÿÿÿÿÿÿÿå
+eeg_classifier_1/conv1d_7/Conv1D/ExpandDims
ExpandDims5eeg_classifier_1/average_pooling1d_1/Squeeze:output:08eeg_classifier_1/conv1d_7/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´ Æ
<eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpEeeg_classifier_1_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:  *
dtype0s
1eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : î
-eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1
ExpandDimsDeeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp:value:0:eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:  ü
 eeg_classifier_1/conv1d_7/Conv1DConv2D4eeg_classifier_1/conv1d_7/Conv1D/ExpandDims:output:06eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
paddingVALID*
strides
µ
(eeg_classifier_1/conv1d_7/Conv1D/SqueezeSqueeze)eeg_classifier_1/conv1d_7/Conv1D:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ *
squeeze_dims

ýÿÿÿÿÿÿÿÿ¦
0eeg_classifier_1/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp9eeg_classifier_1_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ð
!eeg_classifier_1/conv1d_7/BiasAddBiasAdd1eeg_classifier_1/conv1d_7/Conv1D/Squeeze:output:08eeg_classifier_1/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ 
eeg_classifier_1/conv1d_7/ReluRelu*eeg_classifier_1/conv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ 
-eeg_classifier_1/spatial_dropout1d_3/IdentityIdentity,eeg_classifier_1/conv1d_7/Relu:activations:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ q
 eeg_classifier_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà%  Ã
"eeg_classifier_1/flatten_1/ReshapeReshape6eeg_classifier_1/spatial_dropout1d_3/Identity:output:0)eeg_classifier_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK¨
.eeg_classifier_1/dense_4/MatMul/ReadVariableOpReadVariableOp7eeg_classifier_1_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
àK¨*
dtype0Á
eeg_classifier_1/dense_4/MatMulMatMul+eeg_classifier_1/flatten_1/Reshape:output:06eeg_classifier_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¥
/eeg_classifier_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp8eeg_classifier_1_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:¨*
dtype0Â
 eeg_classifier_1/dense_4/BiasAddBiasAdd)eeg_classifier_1/dense_4/MatMul:product:07eeg_classifier_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
eeg_classifier_1/dense_4/ReluRelu)eeg_classifier_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
#eeg_classifier_1/dropout_3/IdentityIdentity+eeg_classifier_1/dense_4/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨¨
.eeg_classifier_1/dense_5/MatMul/ReadVariableOpReadVariableOp7eeg_classifier_1_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
¨*
dtype0Â
eeg_classifier_1/dense_5/MatMulMatMul,eeg_classifier_1/dropout_3/Identity:output:06eeg_classifier_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/eeg_classifier_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp8eeg_classifier_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 eeg_classifier_1/dense_5/BiasAddBiasAdd)eeg_classifier_1/dense_5/MatMul:product:07eeg_classifier_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
eeg_classifier_1/dense_5/ReluRelu)eeg_classifier_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#eeg_classifier_1/dropout_4/IdentityIdentity+eeg_classifier_1/dense_5/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
.eeg_classifier_1/dense_7/MatMul/ReadVariableOpReadVariableOp7eeg_classifier_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Á
eeg_classifier_1/dense_7/MatMulMatMul,eeg_classifier_1/dropout_4/Identity:output:06eeg_classifier_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/eeg_classifier_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp8eeg_classifier_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 eeg_classifier_1/dense_7/BiasAddBiasAdd)eeg_classifier_1/dense_7/MatMul:product:07eeg_classifier_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 eeg_classifier_1/dense_7/SoftmaxSoftmax)eeg_classifier_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
IdentityIdentity*eeg_classifier_1/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«

NoOpNoOp;^eeg_classifier_1/batch_normalization_2/Cast/ReadVariableOp=^eeg_classifier_1/batch_normalization_2/Cast_1/ReadVariableOp=^eeg_classifier_1/batch_normalization_2/Cast_2/ReadVariableOp=^eeg_classifier_1/batch_normalization_2/Cast_3/ReadVariableOp;^eeg_classifier_1/batch_normalization_3/Cast/ReadVariableOp=^eeg_classifier_1/batch_normalization_3/Cast_1/ReadVariableOp=^eeg_classifier_1/batch_normalization_3/Cast_2/ReadVariableOp=^eeg_classifier_1/batch_normalization_3/Cast_3/ReadVariableOp1^eeg_classifier_1/conv1d_4/BiasAdd/ReadVariableOp=^eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp1^eeg_classifier_1/conv1d_5/BiasAdd/ReadVariableOp=^eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp1^eeg_classifier_1/conv1d_6/BiasAdd/ReadVariableOp=^eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp1^eeg_classifier_1/conv1d_7/BiasAdd/ReadVariableOp=^eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp0^eeg_classifier_1/dense_4/BiasAdd/ReadVariableOp/^eeg_classifier_1/dense_4/MatMul/ReadVariableOp0^eeg_classifier_1/dense_5/BiasAdd/ReadVariableOp/^eeg_classifier_1/dense_5/MatMul/ReadVariableOp0^eeg_classifier_1/dense_7/BiasAdd/ReadVariableOp/^eeg_classifier_1/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : 2x
:eeg_classifier_1/batch_normalization_2/Cast/ReadVariableOp:eeg_classifier_1/batch_normalization_2/Cast/ReadVariableOp2|
<eeg_classifier_1/batch_normalization_2/Cast_1/ReadVariableOp<eeg_classifier_1/batch_normalization_2/Cast_1/ReadVariableOp2|
<eeg_classifier_1/batch_normalization_2/Cast_2/ReadVariableOp<eeg_classifier_1/batch_normalization_2/Cast_2/ReadVariableOp2|
<eeg_classifier_1/batch_normalization_2/Cast_3/ReadVariableOp<eeg_classifier_1/batch_normalization_2/Cast_3/ReadVariableOp2x
:eeg_classifier_1/batch_normalization_3/Cast/ReadVariableOp:eeg_classifier_1/batch_normalization_3/Cast/ReadVariableOp2|
<eeg_classifier_1/batch_normalization_3/Cast_1/ReadVariableOp<eeg_classifier_1/batch_normalization_3/Cast_1/ReadVariableOp2|
<eeg_classifier_1/batch_normalization_3/Cast_2/ReadVariableOp<eeg_classifier_1/batch_normalization_3/Cast_2/ReadVariableOp2|
<eeg_classifier_1/batch_normalization_3/Cast_3/ReadVariableOp<eeg_classifier_1/batch_normalization_3/Cast_3/ReadVariableOp2d
0eeg_classifier_1/conv1d_4/BiasAdd/ReadVariableOp0eeg_classifier_1/conv1d_4/BiasAdd/ReadVariableOp2|
<eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp<eeg_classifier_1/conv1d_4/Conv1D/ExpandDims_1/ReadVariableOp2d
0eeg_classifier_1/conv1d_5/BiasAdd/ReadVariableOp0eeg_classifier_1/conv1d_5/BiasAdd/ReadVariableOp2|
<eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp<eeg_classifier_1/conv1d_5/Conv1D/ExpandDims_1/ReadVariableOp2d
0eeg_classifier_1/conv1d_6/BiasAdd/ReadVariableOp0eeg_classifier_1/conv1d_6/BiasAdd/ReadVariableOp2|
<eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp<eeg_classifier_1/conv1d_6/Conv1D/ExpandDims_1/ReadVariableOp2d
0eeg_classifier_1/conv1d_7/BiasAdd/ReadVariableOp0eeg_classifier_1/conv1d_7/BiasAdd/ReadVariableOp2|
<eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp<eeg_classifier_1/conv1d_7/Conv1D/ExpandDims_1/ReadVariableOp2b
/eeg_classifier_1/dense_4/BiasAdd/ReadVariableOp/eeg_classifier_1/dense_4/BiasAdd/ReadVariableOp2`
.eeg_classifier_1/dense_4/MatMul/ReadVariableOp.eeg_classifier_1/dense_4/MatMul/ReadVariableOp2b
/eeg_classifier_1/dense_5/BiasAdd/ReadVariableOp/eeg_classifier_1/dense_5/BiasAdd/ReadVariableOp2`
.eeg_classifier_1/dense_5/MatMul/ReadVariableOp.eeg_classifier_1/dense_5/MatMul/ReadVariableOp2b
/eeg_classifier_1/dense_7/BiasAdd/ReadVariableOp/eeg_classifier_1/dense_7/BiasAdd/ReadVariableOp2`
.eeg_classifier_1/dense_7/MatMul/ReadVariableOp.eeg_classifier_1/dense_7/MatMul/ReadVariableOp:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ü
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_603234

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¨:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
 
_user_specified_nameinputs
Á
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_603210

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿà%  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàKY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿàK"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¯ :T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯ 
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*°
serving_default
@
input_15
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ù
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		batch_n_1
	
conv2
	batch_n_2
spatial_drop_1
	conv3
	avg_pool1
	conv4
spatial_drop_2
flat

dense1
dropout1

dense2
dropout2

dense3
dropout3
out
	optimizer

signatures"
_tf_keras_model
Æ
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11
'12
(13
)14
*15
+16
,17
-18
.19
/20
021"
trackable_list_wrapper
¦
0
1
2
3
!4
"5
#6
$7
'8
(9
)10
*11
+12
,13
-14
.15
/16
017"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ó
6trace_0
7trace_1
8trace_2
9trace_32
1__inference_eeg_classifier_1_layer_call_fn_603325
1__inference_eeg_classifier_1_layer_call_fn_603889
1__inference_eeg_classifier_1_layer_call_fn_603938
1__inference_eeg_classifier_1_layer_call_fn_603657¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z6trace_0z7trace_1z8trace_2z9trace_3
ß
:trace_0
;trace_1
<trace_2
=trace_32ô
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604053
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604244
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603720
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603783¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z:trace_0z;trace_1z<trace_2z=trace_3
ÌBÉ
!__inference__wrapped_model_602831input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ý
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

kernel
bias
 D_jit_compiled_convolution_op"
_tf_keras_layer
ê
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kaxis
	gamma
beta
moving_mean
 moving_variance"
_tf_keras_layer
Ý
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses

!kernel
"bias
 R_jit_compiled_convolution_op"
_tf_keras_layer
ê
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	#gamma
$beta
%moving_mean
&moving_variance"
_tf_keras_layer
¼
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
`_random_generator"
_tf_keras_layer
Ý
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

'kernel
(bias
 g_jit_compiled_convolution_op"
_tf_keras_layer
¥
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses

)kernel
*bias
 t_jit_compiled_convolution_op"
_tf_keras_layer
¼
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_random_generator"
_tf_keras_layer
§
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
)
	keras_api"
_tf_keras_layer
A
	keras_api
_random_generator"
_tf_keras_layer
Á
	variables
 trainable_variables
¡regularization_losses
¢	keras_api
£__call__
+¤&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
À
	¥iter
¦beta_1
§beta_2

¨decay
©learning_ratem«m¬m­m®!m¯"m°#m±$m²'m³(m´)mµ*m¶+m·,m¸-m¹.mº/m»0m¼v½v¾v¿vÀ!vÁ"vÂ#vÃ$vÄ'vÅ(vÆ)vÇ*vÈ+vÉ,vÊ-vË.vÌ/vÍ0vÎ"
	optimizer
-
ªserving_default"
signature_map
6:4 2 eeg_classifier_1/conv1d_4/kernel
,:* 2eeg_classifier_1/conv1d_4/bias
::8 2,eeg_classifier_1/batch_normalization_2/gamma
9:7 2+eeg_classifier_1/batch_normalization_2/beta
B:@  (22eeg_classifier_1/batch_normalization_2/moving_mean
F:D  (26eeg_classifier_1/batch_normalization_2/moving_variance
6:4  2 eeg_classifier_1/conv1d_5/kernel
,:* 2eeg_classifier_1/conv1d_5/bias
::8 2,eeg_classifier_1/batch_normalization_3/gamma
9:7 2+eeg_classifier_1/batch_normalization_3/beta
B:@  (22eeg_classifier_1/batch_normalization_3/moving_mean
F:D  (26eeg_classifier_1/batch_normalization_3/moving_variance
6:4  2 eeg_classifier_1/conv1d_6/kernel
,:* 2eeg_classifier_1/conv1d_6/bias
6:4  2 eeg_classifier_1/conv1d_7/kernel
,:* 2eeg_classifier_1/conv1d_7/bias
3:1
àK¨2eeg_classifier_1/dense_4/kernel
,:*¨2eeg_classifier_1/dense_4/bias
3:1
¨2eeg_classifier_1/dense_5/kernel
,:*2eeg_classifier_1/dense_5/bias
2:0	2eeg_classifier_1/dense_7/kernel
+:)2eeg_classifier_1/dense_7/bias
<
0
 1
%2
&3"
trackable_list_wrapper

0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
«0
¬1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
1__inference_eeg_classifier_1_layer_call_fn_603325input_1"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
1__inference_eeg_classifier_1_layer_call_fn_603889input_tensor"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bÿ
1__inference_eeg_classifier_1_layer_call_fn_603938input_tensor"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
1__inference_eeg_classifier_1_layer_call_fn_603657input_1"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604053input_tensor"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604244input_tensor"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603720input_1"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603783input_1"¹
°²¬
FullArgSpec/
args'$
jself
jinput_tensor

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
ï
²trace_02Ð
)__inference_conv1d_4_layer_call_fn_604253¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z²trace_0

³trace_02ë
D__inference_conv1d_4_layer_call_and_return_conditional_losses_604269¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z³trace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
0
1
2
 3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
â
¹trace_0
ºtrace_12§
6__inference_batch_normalization_2_layer_call_fn_604282
6__inference_batch_normalization_2_layer_call_fn_604295´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z¹trace_0zºtrace_1

»trace_0
¼trace_12Ý
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604315
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604349´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z»trace_0z¼trace_1
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
ï
Âtrace_02Ð
)__inference_conv1d_5_layer_call_fn_604358¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÂtrace_0

Ãtrace_02ë
D__inference_conv1d_5_layer_call_and_return_conditional_losses_604374¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÃtrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
<
#0
$1
%2
&3"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Änon_trainable_variables
Ålayers
Æmetrics
 Çlayer_regularization_losses
Èlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
â
Étrace_0
Êtrace_12§
6__inference_batch_normalization_3_layer_call_fn_604387
6__inference_batch_normalization_3_layer_call_fn_604400´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÉtrace_0zÊtrace_1

Ëtrace_0
Ìtrace_12Ý
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604420
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604454´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zËtrace_0zÌtrace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
Þ
Òtrace_0
Ótrace_12£
4__inference_spatial_dropout1d_2_layer_call_fn_604459
4__inference_spatial_dropout1d_2_layer_call_fn_604464´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÒtrace_0zÓtrace_1

Ôtrace_0
Õtrace_12Ù
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604469
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604491´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zÔtrace_0zÕtrace_1
"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
ï
Ûtrace_02Ð
)__inference_conv1d_6_layer_call_fn_604500¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÛtrace_0

Ütrace_02ë
D__inference_conv1d_6_layer_call_and_return_conditional_losses_604516¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÜtrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
ú
âtrace_02Û
4__inference_average_pooling1d_1_layer_call_fn_604521¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zâtrace_0

ãtrace_02ö
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_604529¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zãtrace_0
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
ï
étrace_02Ð
)__inference_conv1d_7_layer_call_fn_604538¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zétrace_0

êtrace_02ë
D__inference_conv1d_7_layer_call_and_return_conditional_losses_604554¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zêtrace_0
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Þ
ðtrace_0
ñtrace_12£
4__inference_spatial_dropout1d_3_layer_call_fn_604559
4__inference_spatial_dropout1d_3_layer_call_fn_604564´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zðtrace_0zñtrace_1

òtrace_0
ótrace_12Ù
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604569
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604591´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zòtrace_0zótrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ð
ùtrace_02Ñ
*__inference_flatten_1_layer_call_fn_604596¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zùtrace_0

útrace_02ì
E__inference_flatten_1_layer_call_and_return_conditional_losses_604602¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zútrace_0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_dense_4_layer_call_fn_604611¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_dense_4_layer_call_and_return_conditional_losses_604622¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ê
trace_0
trace_12
*__inference_dropout_3_layer_call_fn_604627
*__inference_dropout_3_layer_call_fn_604632´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Å
E__inference_dropout_3_layer_call_and_return_conditional_losses_604637
E__inference_dropout_3_layer_call_and_return_conditional_losses_604649´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_dense_5_layer_call_fn_604658¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_dense_5_layer_call_and_return_conditional_losses_604669¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ê
trace_0
trace_12
*__inference_dropout_4_layer_call_fn_604674
*__inference_dropout_4_layer_call_fn_604679´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Å
E__inference_dropout_4_layer_call_and_return_conditional_losses_604684
E__inference_dropout_4_layer_call_and_return_conditional_losses_604696´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ztrace_0ztrace_1
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
 trainable_variables
¡regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
î
 trace_02Ï
(__inference_dense_7_layer_call_fn_604705¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0

¡trace_02ê
C__inference_dense_7_layer_call_and_return_conditional_losses_604716¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¡trace_0
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ËBÈ
$__inference_signature_wrapper_603840input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
¢	variables
£	keras_api

¤total

¥count"
_tf_keras_metric
c
¦	variables
§	keras_api

¨total

©count
ª
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_conv1d_4_layer_call_fn_604253inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv1d_4_layer_call_and_return_conditional_losses_604269inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_2_layer_call_fn_604282inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_2_layer_call_fn_604295inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604315inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604349inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_conv1d_5_layer_call_fn_604358inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv1d_5_layer_call_and_return_conditional_losses_604374inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
üBù
6__inference_batch_normalization_3_layer_call_fn_604387inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
üBù
6__inference_batch_normalization_3_layer_call_fn_604400inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604420inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604454inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
4__inference_spatial_dropout1d_2_layer_call_fn_604459inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
4__inference_spatial_dropout1d_2_layer_call_fn_604464inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604469inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604491inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_conv1d_6_layer_call_fn_604500inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv1d_6_layer_call_and_return_conditional_losses_604516inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
èBå
4__inference_average_pooling1d_1_layer_call_fn_604521inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_604529inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÝBÚ
)__inference_conv1d_7_layer_call_fn_604538inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
øBõ
D__inference_conv1d_7_layer_call_and_return_conditional_losses_604554inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
úB÷
4__inference_spatial_dropout1d_3_layer_call_fn_604559inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
úB÷
4__inference_spatial_dropout1d_3_layer_call_fn_604564inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604569inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604591inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÞBÛ
*__inference_flatten_1_layer_call_fn_604596inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_flatten_1_layer_call_and_return_conditional_losses_604602inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_4_layer_call_fn_604611inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_4_layer_call_and_return_conditional_losses_604622inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ðBí
*__inference_dropout_3_layer_call_fn_604627inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_3_layer_call_fn_604632inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_3_layer_call_and_return_conditional_losses_604637inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_3_layer_call_and_return_conditional_losses_604649inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_5_layer_call_fn_604658inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_5_layer_call_and_return_conditional_losses_604669inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ðBí
*__inference_dropout_4_layer_call_fn_604674inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ðBí
*__inference_dropout_4_layer_call_fn_604679inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_4_layer_call_and_return_conditional_losses_604684inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
E__inference_dropout_4_layer_call_and_return_conditional_losses_604696inputs"´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_7_layer_call_fn_604705inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_7_layer_call_and_return_conditional_losses_604716inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
0
¤0
¥1"
trackable_list_wrapper
.
¢	variables"
_generic_user_object
:  (2total
:  (2count
0
¨0
©1"
trackable_list_wrapper
.
¦	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
;:9 2'Adam/eeg_classifier_1/conv1d_4/kernel/m
1:/ 2%Adam/eeg_classifier_1/conv1d_4/bias/m
?:= 23Adam/eeg_classifier_1/batch_normalization_2/gamma/m
>:< 22Adam/eeg_classifier_1/batch_normalization_2/beta/m
;:9  2'Adam/eeg_classifier_1/conv1d_5/kernel/m
1:/ 2%Adam/eeg_classifier_1/conv1d_5/bias/m
?:= 23Adam/eeg_classifier_1/batch_normalization_3/gamma/m
>:< 22Adam/eeg_classifier_1/batch_normalization_3/beta/m
;:9  2'Adam/eeg_classifier_1/conv1d_6/kernel/m
1:/ 2%Adam/eeg_classifier_1/conv1d_6/bias/m
;:9  2'Adam/eeg_classifier_1/conv1d_7/kernel/m
1:/ 2%Adam/eeg_classifier_1/conv1d_7/bias/m
8:6
àK¨2&Adam/eeg_classifier_1/dense_4/kernel/m
1:/¨2$Adam/eeg_classifier_1/dense_4/bias/m
8:6
¨2&Adam/eeg_classifier_1/dense_5/kernel/m
1:/2$Adam/eeg_classifier_1/dense_5/bias/m
7:5	2&Adam/eeg_classifier_1/dense_7/kernel/m
0:.2$Adam/eeg_classifier_1/dense_7/bias/m
;:9 2'Adam/eeg_classifier_1/conv1d_4/kernel/v
1:/ 2%Adam/eeg_classifier_1/conv1d_4/bias/v
?:= 23Adam/eeg_classifier_1/batch_normalization_2/gamma/v
>:< 22Adam/eeg_classifier_1/batch_normalization_2/beta/v
;:9  2'Adam/eeg_classifier_1/conv1d_5/kernel/v
1:/ 2%Adam/eeg_classifier_1/conv1d_5/bias/v
?:= 23Adam/eeg_classifier_1/batch_normalization_3/gamma/v
>:< 22Adam/eeg_classifier_1/batch_normalization_3/beta/v
;:9  2'Adam/eeg_classifier_1/conv1d_6/kernel/v
1:/ 2%Adam/eeg_classifier_1/conv1d_6/bias/v
;:9  2'Adam/eeg_classifier_1/conv1d_7/kernel/v
1:/ 2%Adam/eeg_classifier_1/conv1d_7/bias/v
8:6
àK¨2&Adam/eeg_classifier_1/dense_4/kernel/v
1:/¨2$Adam/eeg_classifier_1/dense_4/bias/v
8:6
¨2&Adam/eeg_classifier_1/dense_5/kernel/v
1:/2$Adam/eeg_classifier_1/dense_5/bias/v
7:5	2&Adam/eeg_classifier_1/dense_7/kernel/v
0:.2$Adam/eeg_classifier_1/dense_7/bias/vª
!__inference__wrapped_model_602831 !"%&$#'()*+,-./05¢2
+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿØ
O__inference_average_pooling1d_1_layer_call_and_return_conditional_losses_604529E¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
4__inference_average_pooling1d_1_layer_call_fn_604521wE¢B
;¢8
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÑ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604315| @¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_604349| @¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ©
6__inference_batch_normalization_2_layer_call_fn_604282o @¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_2_layer_call_fn_604295o @¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ñ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604420|%&$#@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ñ
Q__inference_batch_normalization_3_layer_call_and_return_conditional_losses_604454|%&$#@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ©
6__inference_batch_normalization_3_layer_call_fn_604387o%&$#@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ©
6__inference_batch_normalization_3_layer_call_fn_604400o%&$#@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ®
D__inference_conv1d_4_layer_call_and_return_conditional_losses_604269f4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ 
 
)__inference_conv1d_4_layer_call_fn_604253Y4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ ®
D__inference_conv1d_5_layer_call_and_return_conditional_losses_604374f!"4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿí 
 
)__inference_conv1d_5_layer_call_fn_604358Y!"4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿí ®
D__inference_conv1d_6_layer_call_and_return_conditional_losses_604516f'(4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿí 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿè 
 
)__inference_conv1d_6_layer_call_fn_604500Y'(4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿí 
ª "ÿÿÿÿÿÿÿÿÿè ®
D__inference_conv1d_7_layer_call_and_return_conditional_losses_604554f)*4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ´ 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¯ 
 
)__inference_conv1d_7_layer_call_fn_604538Y)*4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ´ 
ª "ÿÿÿÿÿÿÿÿÿ¯ ¥
C__inference_dense_4_layer_call_and_return_conditional_losses_604622^+,0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿàK
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¨
 }
(__inference_dense_4_layer_call_fn_604611Q+,0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿàK
ª "ÿÿÿÿÿÿÿÿÿ¨¥
C__inference_dense_5_layer_call_and_return_conditional_losses_604669^-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¨
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_5_layer_call_fn_604658Q-.0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ¨
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_7_layer_call_and_return_conditional_losses_604716]/00¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense_7_layer_call_fn_604705P/00¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_dropout_3_layer_call_and_return_conditional_losses_604637^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ¨
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¨
 §
E__inference_dropout_3_layer_call_and_return_conditional_losses_604649^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ¨
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ¨
 
*__inference_dropout_3_layer_call_fn_604627Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ¨
p 
ª "ÿÿÿÿÿÿÿÿÿ¨
*__inference_dropout_3_layer_call_fn_604632Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ¨
p
ª "ÿÿÿÿÿÿÿÿÿ¨§
E__inference_dropout_4_layer_call_and_return_conditional_losses_604684^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_4_layer_call_and_return_conditional_losses_604696^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_4_layer_call_fn_604674Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_4_layer_call_fn_604679Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÊ
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603720z !"%&$#'()*+,-./09¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_603783z !"%&$#'()*+,-./09¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604053 !"%&$#'()*+,-./0>¢;
4¢1
+(
input_tensorÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ï
L__inference_eeg_classifier_1_layer_call_and_return_conditional_losses_604244 !"%&$#'()*+,-./0>¢;
4¢1
+(
input_tensorÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¢
1__inference_eeg_classifier_1_layer_call_fn_603325m !"%&$#'()*+,-./09¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¢
1__inference_eeg_classifier_1_layer_call_fn_603657m !"%&$#'()*+,-./09¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
1__inference_eeg_classifier_1_layer_call_fn_603889r !"%&$#'()*+,-./0>¢;
4¢1
+(
input_tensorÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ§
1__inference_eeg_classifier_1_layer_call_fn_603938r !"%&$#'()*+,-./0>¢;
4¢1
+(
input_tensorÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ§
E__inference_flatten_1_layer_call_and_return_conditional_losses_604602^4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¯ 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿàK
 
*__inference_flatten_1_layer_call_fn_604596Q4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¯ 
ª "ÿÿÿÿÿÿÿÿÿàK¸
$__inference_signature_wrapper_603840 !"%&$#'()*+,-./0@¢=
¢ 
6ª3
1
input_1&#
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿÜ
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604469I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ü
O__inference_spatial_dropout1d_2_layer_call_and_return_conditional_losses_604491I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
4__inference_spatial_dropout1d_2_layer_call_fn_604459{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
4__inference_spatial_dropout1d_2_layer_call_fn_604464{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÜ
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604569I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ü
O__inference_spatial_dropout1d_3_layer_call_and_return_conditional_losses_604591I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";¢8
1.
0'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ³
4__inference_spatial_dropout1d_3_layer_call_fn_604559{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
4__inference_spatial_dropout1d_3_layer_call_fn_604564{I¢F
?¢<
63
inputs'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ".+'ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ