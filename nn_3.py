import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import struct
import torch.nn.functional as F

def dec_to_bin_list(dec_number, num_bits):
    if isinstance(dec_number, torch.Tensor):
        if dec_number.numel() > 1:
            # Handle batch of tensors
            return [dec_to_bin_list(item.item(), num_bits) for item in dec_number]
        else:
            # Convert single tensor element to Python scalar
            dec_number = int(dec_number.item())
    else:
        dec_number = int(dec_number)  # Convert to integer if it's a float or integer
    
    # Handle negative numbers by taking absolute value before converting to binary
    binary_str = bin(abs(dec_number))[2:].zfill(num_bits) 
    return [int(bit) for bit in binary_str]

def bin_list_to_decimal(bin_list):
    if isinstance(bin_list[0], list):
        return [int(''.join(map(str, bits)), 2) for bits in bin_list]
    return int(''.join(map(str, bin_list)), 2)


def binary_addition(a, b, c, d, carry_in):
    total = a + b + c + d + carry_in
    result = total % 2
    carry_out = total // 2
    return carry_out, result


def binary_addition1(a, b, carry_in):
    total = a + b + carry_in
    result = total % 2
    carry_out = total // 2
    return carry_out, result

def compressor_1(a, b, c, d):
    p1 = a ^ b
    p2 = c ^ d
    sum_ = p1 ^ p2
    carry = (p1 & p2) | (a & b)
    return sum_, carry

def compressor_2(a, b, c, d, carry_in):
    sum2 = a ^ b ^ c ^ d ^ carry_in
    carry_out = (a & b) | (b & c) | (c & a) | (a & d) | (b & d) | (c & d)
    return sum2 
    
def shift_and_pad(bits, shift_amount, total_length):
    return [0] * shift_amount + bits + [0] * (total_length - len(bits) - shift_amount)


def appx_multiplier8x8_tensor(a, b):
    
    ALXL = [[0]*8 for _ in range(4)]
    ALXH = [[0]*8 for _ in range(4)]
    AHXL = [[0]*8 for _ in range(4)]

    ALXL_result = [0]*8
    ALXH_result = [0]*8
    AHXL_result = [0]*8

    result = [0]*16

    A_binary = dec_to_bin_list(a, 8)
    B_binary = dec_to_bin_list(b, 8)

    ALXL[0][4:] = [A_binary[4] & B_binary[7], A_binary[5] & B_binary[7], A_binary[6] & B_binary[7], A_binary[7] & B_binary[7]]
    ALXL[1][3:7] = [A_binary[4] & B_binary[6], A_binary[5] & B_binary[6], A_binary[6] & B_binary[6], A_binary[7] & B_binary[6]]
    ALXL[2][2:6] = [A_binary[4] & B_binary[5], A_binary[5] & B_binary[5], A_binary[6] & B_binary[5], A_binary[7] & B_binary[5]]
    ALXL[3][1:5] = [A_binary[4] & B_binary[4], A_binary[5] & B_binary[4], A_binary[6] & B_binary[4], A_binary[7] & B_binary[4]]

        
       
    AHXL[0][4:] = [A_binary[0] & B_binary[7], A_binary[1] & B_binary[7], A_binary[2] & B_binary[7], A_binary[3] & B_binary[7]]
    AHXL[1][3:7] = [A_binary[0] & B_binary[6], A_binary[1] & B_binary[6], A_binary[2] & B_binary[6], A_binary[3] & B_binary[6]]
    AHXL[2][2:6] = [A_binary[0] & B_binary[5], A_binary[1] & B_binary[5], A_binary[2] & B_binary[5], A_binary[3] & B_binary[5]]
    AHXL[3][1:5] = [A_binary[0] & B_binary[4], A_binary[1] & B_binary[4], A_binary[2] & B_binary[4], A_binary[3] & B_binary[4]]

        

    ALXH[0][4:] = [A_binary[4] & B_binary[3], A_binary[5] & B_binary[3], A_binary[6] & B_binary[3], A_binary[7] & B_binary[3]]
    ALXH[1][3:7] = [A_binary[4] & B_binary[2], A_binary[5] & B_binary[2], A_binary[6] & B_binary[2], A_binary[7] & B_binary[2]]
    ALXH[2][2:6] = [A_binary[4] & B_binary[1], A_binary[5] & B_binary[1], A_binary[6] & B_binary[1], A_binary[7] & B_binary[1]]
    ALXH[3][1:5] = [A_binary[4] & B_binary[0], A_binary[5] & B_binary[0], A_binary[6] & B_binary[0], A_binary[7] & B_binary[0]]

    
     
    carry_in_ALXL = (ALXL[0][4] & ALXL[1][4]) | ALXL[2][4] | ALXL[3][4]
    carry_in_ALXH = (ALXH[0][4] & ALXH[1][4]) | ALXH[2][4] | ALXH[3][4]
    carry_in_AHXL = (AHXL[0][4] & AHXL[1][4]) | AHXL[2][4] | AHXL[3][4]      

    

     # Calculate ALXL_result
    ALXL_result[7] = ALXL[0][7]| ALXL[1][7]| ALXL[2][7]| ALXL[3][7]
    ALXL_result[6] = ALXL[0][6]| ALXL[1][6]| ALXL[2][6]| ALXL[3][6]
    ALXL_result[5], carry = compressor_1(ALXL[0][5], ALXL[1][5], ALXL[2][5], ALXL[3][5])
    ALXL_result[4] = compressor_2(ALXL[0][4], ALXL[1][4], ALXL[2][4], ALXL[3][4], carry)
    intermediate_carry1 = [0]*3
    intermediate_carry1[0], ALXL_result[3] = binary_addition(ALXL[0][3], ALXL[1][3], ALXL[2][3], ALXL[3][3], 0)
    intermediate_carry1[1], ALXL_result[2] = binary_addition(ALXL[0][2], ALXL[1][2], ALXL[2][2], ALXL[3][2], intermediate_carry1[0])
    intermediate_carry1[2], ALXL_result[1] = binary_addition(ALXL[0][1], ALXL[1][1], ALXL[2][1], ALXL[3][1], intermediate_carry1[1])
    _, ALXL_result[0] = binary_addition(ALXL[0][0], ALXL[1][0], ALXL[2][0], ALXL[3][0], intermediate_carry1[2])



    # Calculate ALXH_result
    ALXH_result[7] = ALXH[0][7]| ALXH[1][7]| ALXH[2][7]| ALXH[3][7]
    ALXH_result[6] = ALXH[0][6]| ALXH[1][6]| ALXH[2][6]| ALXH[3][6]
    ALXH_result[5], carry = compressor_1(ALXH[0][5], ALXH[1][5], ALXH[2][5], ALXH[3][5])
    ALXH_result[4] = compressor_2(ALXH[0][4], ALXH[1][4], ALXH[2][4], ALXH[3][4], carry)
    intermediate_carry2 = [0]*3
    intermediate_carry2[0], ALXH_result[3] = binary_addition(ALXH[0][3], ALXH[1][3], ALXH[2][3], ALXH[3][3], 0)
    intermediate_carry2[1], ALXH_result[2] = binary_addition(ALXH[0][2], ALXH[1][2], ALXH[2][2], ALXH[3][2], intermediate_carry2[0])
    intermediate_carry2[2], ALXH_result[1] = binary_addition(ALXH[0][1], ALXH[1][1], ALXH[2][1], ALXH[3][1], intermediate_carry2[1])
    _, ALXH_result[0] = binary_addition(ALXH[0][0], ALXH[1][0], ALXH[2][0], ALXH[3][0], intermediate_carry2[2])

    

    # Calculate AHXL_result
    AHXL_result[7] = AHXL[0][7]| AHXL[1][7]| AHXL[2][7]| AHXL[3][7]
    AHXL_result[6] = AHXL[0][6]| AHXL[1][6]| AHXL[2][6]| AHXL[3][6]
    AHXL_result[5], carry = compressor_1(AHXL[0][5], AHXL[1][5], AHXL[2][5], AHXL[3][5])
    AHXL_result[4] = compressor_2(AHXL[0][4], AHXL[1][4], AHXL[2][4], AHXL[3][4], carry)
    intermediate_carry3 = [0]*3
    intermediate_carry3[0], AHXL_result[3] = binary_addition(AHXL[0][3], AHXL[1][3], AHXL[2][3], AHXL[3][3], 0)
    intermediate_carry3[1], AHXL_result[2] = binary_addition(AHXL[0][2], AHXL[1][2], AHXL[2][2], AHXL[3][2], intermediate_carry3[0])
    intermediate_carry3[2], AHXL_result[1] = binary_addition(AHXL[0][1], AHXL[1][1], AHXL[2][1], AHXL[3][1], intermediate_carry3[1])
    _, AHXL_result[0] = binary_addition(AHXL[0][0], AHXL[1][0], AHXL[2][0], AHXL[3][0], intermediate_carry3[2])

    AHHXHH=[[0]*8 for _ in range(2)]
    AHHXHL=[[0]*8 for _ in range(2)]
    AHLXHH=[[0]*8 for _ in range(2)]
    AHLXHL=[[0]*8 for _ in range(2)]

    AHHXHH_result = [0]*8
    AHHXHL_result = [0]*8
    AHLXHH_result = [0]*8
    AHLXHL_result = [0]*8

    result1 = [0]*8
    
    A_binary = dec_to_bin_list(a, 8)
    B_binary = dec_to_bin_list(b, 8)

    AHHXHH[0][2:4] = [A_binary[0] & B_binary[1], A_binary[1] & B_binary[1]]
    AHHXHH[1][1:3] = [A_binary[0] & B_binary[0], A_binary[1] & B_binary[0]]

    AHHXHL[0][4:6] = [A_binary[0] & B_binary[3], A_binary[1] & B_binary[3]]
    AHHXHL[1][3:5] = [A_binary[0] & B_binary[2], A_binary[1] & B_binary[2]]

    AHLXHH[0][4:6] = [A_binary[2] & B_binary[1], A_binary[3] & B_binary[1]]
    AHLXHH[1][3:5] = [A_binary[2] & B_binary[0], A_binary[3] & B_binary[0]]

    AHLXHL[0][6:] = [A_binary[2] & B_binary[3], A_binary[3] & B_binary[3]]
    AHLXHL[1][5:7] = [A_binary[2] & B_binary[2], A_binary[3] & B_binary[2]]
   
    carry_HH, AHHXHH_result[3]= binary_addition1(AHHXHH[0][3] , AHHXHH[1][3],0)
    carry_HH1, AHHXHH_result[2]= binary_addition1(AHHXHH[0][2] , AHHXHH[1][2],carry_HH)
    carry_HH2, AHHXHH_result[1]= binary_addition1(AHHXHH[0][1] , AHHXHH[1][1],carry_HH1)
    _, AHHXHH_result[0]= binary_addition1(AHHXHH[0][0] , AHHXHH[1][0],carry_HH2)

    carry_HL, AHHXHL_result[5]= binary_addition1(AHHXHL[0][5] , AHHXHL[1][5],0)
    carry_HL1, AHHXHL_result[4]= binary_addition1(AHHXHL[0][4] , AHHXHL[1][4],carry_HL)
    carry_HL2, AHHXHL_result[3]= binary_addition1(AHHXHL[0][3] , AHHXHL[1][3],carry_HL1)
    _, AHHXHL_result[2]= binary_addition1(AHHXHL[0][2] , AHHXHL[1][2],carry_HL2)
    
    carry_LH, AHLXHH_result[5]= binary_addition1(AHLXHH[0][5] , AHLXHH[1][5],0)
    carry_LH1, AHLXHH_result[4]= binary_addition1(AHLXHH[0][4] , AHLXHH[1][4],carry_LH)
    carry_LH2, AHLXHH_result[3]= binary_addition1(AHLXHH[0][3] , AHLXHH[1][3],carry_LH1)
    _, AHLXHH_result[2]= binary_addition1(AHLXHH[0][2] , AHLXHH[1][2],carry_LH2)
    
    carry_LL,AHLXHL_result[7]= binary_addition1(AHLXHL[0][7] , AHLXHL[1][7],0)
    carry_LL1,AHLXHL_result[6]= binary_addition1(AHLXHL[0][6] , AHLXHL[1][6],carry_LL)
    carry_LL2,AHLXHL_result[5]= binary_addition1(AHLXHL[0][5] , AHLXHL[1][5],carry_LL1)
    _,AHLXHL_result[4]= binary_addition1(AHLXHL[0][4] , AHLXHL[1][4],carry_LL2)

    carry_r1_7, result1[7] = binary_addition(AHHXHH_result[7],AHLXHH_result[7],AHHXHL_result[7], AHLXHL_result[7], 0)
    carry_r1_6, result1[6] = binary_addition(AHHXHH_result[6],AHLXHH_result[6],AHHXHL_result[6], AHLXHL_result[6], carry_r1_7)
    carry_r1_5, result1[5] = binary_addition(AHHXHH_result[5],AHLXHH_result[5],AHHXHL_result[5], AHLXHL_result[5], carry_r1_6)
    carry_r1_4, result1[4] = binary_addition(AHHXHH_result[4],AHLXHH_result[4],AHHXHL_result[4], AHLXHL_result[4], carry_r1_5)
    carry_r1_3, result1[3] = binary_addition(AHHXHH_result[3],AHLXHH_result[3],AHHXHL_result[3], AHLXHL_result[3], carry_r1_4)
    carry_r1_2, result1[2] = binary_addition(AHHXHH_result[2],AHLXHH_result[2],AHHXHL_result[2], AHLXHL_result[2], carry_r1_3)
    carry_r1_1, result1[1] = binary_addition(AHHXHH_result[1],AHLXHH_result[1],AHHXHL_result[1], AHLXHL_result[1], carry_r1_2)
    _, result1[0] = binary_addition(AHHXHH_result[0],AHLXHH_result[0],AHHXHL_result[0], AHLXHL_result[0], carry_r1_1)
    
    
    shifted_result1 = shift_and_pad(result1, 0, 16)
    shifted_ALXL_result = shift_and_pad(ALXL_result, 8, 16)
    shifted_ALXH_result = shift_and_pad(ALXH_result, 4, 16)
    shifted_AHXL_result = shift_and_pad(AHXL_result, 4, 16)

    # Initialize result and carry
    result = [0]*16
    carry = 0

    # Perform binary addition
    for i in range(15, -1, -1):
        sum_ = shifted_result1[i] + shifted_ALXL_result[i] + shifted_ALXH_result[i] + shifted_AHXL_result[i] + carry
        result[i] = sum_ % 2
        carry = sum_ // 2

    # Convert result from binary list to decimal
    final_sum = torch.tensor(bin_list_to_decimal(result), dtype=torch.float32)

    return final_sum

def float_to_bin(num):
    return bin(struct.unpack('!I', struct.pack('!f', num))[0])[2:].zfill(32)

def bin_to_float(b):
    return struct.unpack('!f', struct.pack('!I', int(b, 2)))[0]

def FP_appx_mul(x, y):
    bin_x = float_to_bin(x)
    bin_y = float_to_bin(y)
    
    sign_x, exp_x, mant_x = int(bin_x[0], 2), int(bin_x[1:9], 2), bin_x[9:]
    sign_y, exp_y, mant_y = int(bin_y[0], 2), int(bin_y[1:9], 2), bin_y[9:]
    
    sign_result = sign_x ^ sign_y
    exp_result = exp_x + exp_y - 127
    mant_result = appx_multiplier8x8_tensor(int(mant_x, 2), int(mant_y, 2))
    
    result_bin = f"{sign_result:b}{exp_result:08b}{mant_result:023b}"
    result_float = bin_to_float(result_bin)
    return result_float

class ApproxMultLayer(nn.Module):
    def forward(self, x, y):
        return FP_appx_mul(x.item(), y.item())

# Define the neural network architecture
class SimpleCNN(nn.Module):
    def __init__(self): # Fix: Correct the typo _init to __init__
        super(SimpleCNN, self).__init__() # Fix: Call super().__init__() to initialize layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.approx_mult = ApproxMultLayer()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
accuracy = 100. * correct / len(test_loader.dataset)
print(f'Test loss: {test_loss}, Accuracy: {accuracy}%')