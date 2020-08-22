'''
Funções referentes as operações de visão computacional e algumas auxiliares.

Programador: Renan Sandes
Data: Mai/2020
'''

import time, datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt

class MyImage:
    '''
    classe para auxiliar na manipulação das imagens
    '''
    def __init__(self, img_matrix:np.ndarray):

        # Matriz da imagem
        self.data = img_matrix

        if len(self.data.shape) == 3:
            self.type = 'cor'
        elif len(self.data.shape) == 2:
            self.type = 'tons_de_ciza'
        else:
            self.type = ''


def abrir_img(path):
    '''
    Função para abrir as imagens
    path: diretório da imagem
    returns: ndarray (imagem em matriz)
    '''
    return cv2.imread(path, cv2.IMREAD_COLOR)
    

def normalize_uint8(img):
    '''
    Função para normalizar a imagem. 
    '''
    # Encontrar o máx.
    mx = np.max(img)
    # Encontrar o min
    mn = np.min(img)
    # Nova matriz normalizada.
    # Checar os valores de min e max. Se forem iguais, geram erros.
    if mn != mx:
        img2 = 255 * (img - mn) / (mx-mn)
        return np.uint8(img2)   
    else:
        return np.uint8(img)
    # Retornar convertendo para uint8.
    return np.uint8(img2)


def clip_uint8(img):
    '''Função para clipar os valores na faixa de 8 bits (entre 0 e 255). Retorna img em uint8'''
    img[img>=255] = 255
    img[img<= 0] = 0
    return np.uint8(img)


def convert_color_para_pb(img_color, conv_BGR= np.array([0.1140, 0.5870, 0.2989]) ):
    ''' 
    Converter uma imagem colorida para tons de cinza:
    img_color: B G R
    conv_BGR: Proporção das cores. default = [0.1140, 0.5870, 0.2989]
    '''
    if len(img_color.shape)==3:
        # Converter
        img_bw = img_color[:, :, 0] * conv_BGR[0] + img_color[:, :, 1] * conv_BGR[1] + img_color[:, :, 2] * conv_BGR[2]
        # Normalizar e retornar imangem
        return np.uint8(img_bw)
    
    else:
        return np.uint8(img_color)


def convert_pb_para_bin(img_pb, thresh):
    '''
    Converter imagem em pb para binario
    img_pb: numpy ndarray
    thresh: int entre 0 e 255
    '''
    # Aplicar o limiar
    img_bin = 255 * (img_pb > thresh)
    # Retornar normalizando
    return np.uint8(img_bin)


def soma_imagens(img1:np.ndarray, img2:np.ndarray, normalize=True):
    '''
    Função para somar duas imagens. 
    img1, img2: np.ndarray do mesmo tamanho.
    retorna: np.ndarray normalizada ou clipada.

    '''
    imgA = np.float32(img1)
    imgB = np.float32(img2)

    # Checar os tamanhos da mensagem:
    if (np.array_equal(imgA.shape, imgB.shape)):
        # Caso as imagens sejam do mesmo tamanho:
        # Somar as imagens
        imgC = imgA + imgB

        if normalize:
            # Retornar a matriz normalizada.
            return normalize_uint8(imgC)
        else:
            # Retornar a matriz clipada
            return clip_uint8(imgC)
    else:
        # Jeito besta de retornar se houve algum erro.
        print('Imagens de dimensões diferentes')


def soma_imagem_constante(img ,k, normalize=False):
    '''
    somar uma imagem com uma constante k.
    caso dê valor acima de 255, o valor será clipado para 255.
    '''
    # Somar com a constante.
    imgC = np.float32(img) + k
    # Retornar em uint8
    if normalize:
        return normalize_uint8(imgC)
    else:
        return clip_uint8(imgC)


def multiplicar_imagem_constante(img, k:float, normalize= True):
    ''' Multiplicar uma imagem com uma constante. '''
    imgC = k * img
    if normalize:
        return normalize_uint8(imgC)
    if not normalize:
        return clip_uint8(imgC)


def op_logica_and(imgA:np.ndarray, imgB:np.ndarray):
    '''Realizar a operação lógica AND'''
    # Comparar os tamanhos:
    if np.array_equal(imgA.shape, imgB.shape):
        # Retornar a op and.
        return np.uint8(255 * np.logical_and(imgA>127, imgB>127))


def op_logica_or(imgA:np.ndarray, imgB:np.ndarray):
    '''Realizar a operação lógica OR'''
    # Comparar os tamanhos:
    if np.array_equal(imgA.shape, imgB.shape):
        # Retornar a op or.
        return np.uint8(255 * np.logical_or(imgA>127, imgB>127))


def op_logica_xor(imgA:np.ndarray, imgB:np.ndarray):
    '''Realizar a operação lógica XOR'''
    # Comparar os tamanhos:
    if np.array_equal(imgA.shape, imgB.shape):
        # Retornar a op xor.
        return np.uint8(255 * np.logical_xor(imgA>127, imgB>127))


def op_logica_not(imgA:np.ndarray):
    '''Realizar a operação lógica NOT'''
    # Retornar a op and.
    return np.uint8(255 * np.logical_not(imgA>127))


def op_bitwise_and(imgA:np.ndarray, imgB:np.ndarray):
    '''Realizar a operação lógica bitwise AND'''
    if np.array_equal(imgA.shape, imgB.shape):
        # Retornar a op bitwise and.
        return np.uint8(np.bitwise_and(imgA, imgB))


def op_bitwise_or(imgA:np.ndarray, imgB:np.ndarray):
    '''Realizar a operação lógica bitwise OR'''
    if np.array_equal(imgA.shape, imgB.shape):
        # Retornar a op bitwise or.
        return np.uint8(np.bitwise_or(imgA, imgB))


def op_bitwise_xor(imgA:np.ndarray, imgB:np.ndarray):
    '''Realizar a operação lógica bitwise XOR'''
    if np.array_equal(imgA.shape, imgB.shape):
        # Retornar a op bitwise xor.
        return np.uint8(np.bitwise_xor(imgA, imgB))


def op_bitwise_not(imgA:np.ndarray):
    '''Realizar a operação lógica bitwise NOT'''
    # Retornar a op bitwise and.
    return np.uint8(np.bitwise_not(imgA))


class my_T:
    '''
    Classe com os métodos para retornar as matrizes para transformações geométricas
    '''
    @staticmethod
    def traslation(dx, dy):
        return np.array([[1, 0, dy],[0, 1, dx], [0, 0, 1]])

    @staticmethod
    def rotation(theta, unit='rad'):
        '''
        theta: angle of rotation
        unit: 'deg' or 'rad'. 'rad' by default
        '''
        if unit == 'deg':
            theta = np.pi* theta / 180
 
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [0            ,  0            , 1]])

    @staticmethod
    def scale(factor):
        return np.array([[factor, 0, 0],[0, factor, 0], [0, 0, 1]])
    
    @staticmethod
    def shear(sx=0, sy=0):
        '''
        sx for horizontal
        sy for vertical
        
        '''
        return np.array([[1, sy, 0],[sx, 1, 0], [0, 0, 1]])


def transform_geom(img: np.ndarray, dx=0, dy=0, theta=45,scale_factor= 1,center=True,cx=0, cy=0): 
    '''
    Aplicar transformações geométricas a imagens.
    '''
    #t1 =time.perf_counter()

    # Alocar o espaço para a nova imagem.
    imgB = np.zeros(img.shape)

    # Translação
    T = my_T.traslation(dx, dy)
    
    # Rotação
    R = my_T.rotation(theta, unit='deg')
    
    # Escala
    S = my_T.scale(scale_factor)

    # Matriz resultante.    
    T_R = np.linalg.multi_dot([T, R, S])

    if center:
        T_R = np.linalg.multi_dot([
              my_T.traslation(np.floor(img.shape[1]/2), np.floor(img.shape[0]/2)),
              T_R,
              my_T.traslation(-np.floor(img.shape[1]/2), -np.floor(img.shape[0]/2))])

    elif (not center) and (cx !=0 or cy !=0):
        T_R = np.linalg.multi_dot([
              my_T.traslation(cy, cx),
              T_R,
              my_T.traslation(-cy,-cx)])
        

    # Inverter a matriz para realizar busca inversa,
    T_R_ = np.linalg.inv(T_R)
    s = img.shape
    print(s)
    for x in range(s[0]):
        for y in range(s[1]):
            P_l = np.uint16(np.dot(T_R_, np.array([[x], [y], [1]])))
            if 0 < P_l[0] < s[0] and 0 < P_l[1] < s[1]:
                imgB[x, y] = img[P_l[0], P_l[1]]

    #t2 = time.perf_counter()
    #print('Time elapsed: ', t2-t1)

    return np.uint8(imgB)


def transform_geom_rapida(img: np.ndarray, dx=0, dy=0, theta=45,scale_factor= 1,center=True, cx=0, cy=0):
    '''
    Função para realizara as transfomrações geométricas nas imagens
    
    retorna: np.ndarray de mesma dimensão de img.
    
    argumentos:
        - img: imagem a ser transformada
        - dx e dy: int - distancia para translação
        - theta: float -angulo de rotacao
        - scale_factor: float - fator de escala: ampliar ou reduzir
        - center: bool- operar em torno do centro da imagem.
        - cx, cy: int - Caso o centro não seja a origem ou o centro da imagem.
        -

    '''
    # Benchmark:
    #t1 =time.perf_counter()
    # Alocar o espaço para a nova imagem.
    imgB = np.zeros(img.shape)

    # Translação
    T = my_T.traslation(dx, dy)
    
    # Rotação
    R = my_T.rotation(theta, unit='deg')
    
    # Escala
    S = my_T.scale(scale_factor)

    # Matriz resultante.    
    T_R = np.linalg.multi_dot([T, R, S])

    if center:
        T_R = np.linalg.multi_dot([
              my_T.traslation(np.floor(img.shape[1]/2), np.floor(img.shape[0]/2)),
              T_R,
              my_T.traslation(-np.floor(img.shape[1]/2), -np.floor(img.shape[0]/2))])

    elif (not center) and (cx !=0 or cy !=0):
        T_R = np.linalg.multi_dot([
              my_T.traslation(cy, cx),
              T_R,
              my_T.traslation(-cy,-cx)])

    # Inverter a matriz para realizar busca inversa,
    T_R_ = np.linalg.inv(T_R)

    # Dividir em imagens menores
    h_division = 1000
    v_division = 1000
    ys  = np.arange(0, img.shape[0], v_division)
    xs = np.arange(0, img.shape[1], h_division)
    
    # Executar nas subimagens para poupar memória
    for x0 in xs:
        for y0 in ys:

            x1 = x0+ h_division 
            y1 = y0+ v_division

            if x1 > img.shape[1]:
                x1 = img.shape[1]

            if y1 > img.shape[0]:
                y1 = img.shape[0]
    
            # Meshgrid para combinar os pares ordenados de todas os elementos da imagem
            xx, yy = np.meshgrid(np.arange(x0, x1),
                                 np.arange(y0, y1))

            # Criar o array das posições para busca INVERSA!. uint16 para poupar memoria
            P = np.array([yy.flatten(), 
                          xx.flatten(), 
                          np.ones(xx.shape).flatten()]  ,dtype='uint16')

            # Aplicar a transformação
            P_ = np.dot(T_R_, P)

            # Extrair os indices
            P = np.uint16(P[0:2, :])
            P_ =np.uint16(P_[0:2 , :])

            # Condição para retirar os ídices inválidos: Os valores forem maiores que o da imagem. Como são em uint16 não serão negativos.
            valid_index = np.logical_not(
                np.logical_or( P_[0, :]>= img.shape[0],
                               P_[1, :]>= img.shape[1]))    
            
            # Retirar os indices invalidos
            P =  P[:, valid_index] 
            P_= P_[:, valid_index]

            imgB[P[0,:], P[1,:]] = img[P_[0,:], P_[1,:]]
        

    #t2 = time.perf_counter()
    #print('Time elapsed: ', t2-t1)

    return np.uint8(imgB)


def sobel(A:np.ndarray, threshold=127, normalizado=True):
    '''
    Aplica a mascara de sobel aos 
    utilizando poucas iterações.
    
    A: imagem em np.ndarray com apenas 1 canal.
    threshold: Após a aplicação das mascaras threshold.
    normalizado: Pode escolher receber a imagem como float, sem normalização.

    '''

    # Benchmarking
    #t1 = time.perf_counter()
    if len(A.shape)>2:
        A = convert_color_para_pb(A)

    # Alocar espaço para as matrizes:
    By = np.zeros(A.shape)
    Bx = np.zeros(A.shape)

    # Kernels de Sobel
    mask_sx = 0.25 * np.array([[ 1, 0, -1], 
                               [ 2, 0, -2],
                               [ 1, 0, -1]])

    mask_sy = 0.25 * np.array([[-1,-2,-1], 
                               [ 0, 0, 0],
                               [ 1, 2, 1]])

    m = 1; k = 3 # dimensões da mascara
    n = 1; l = 3 # dimensões da mascara

    w = A.shape[1] # Dimensões da imagem
    h = A.shape[0] # Dimensões da imagem

    # Aplicar mascara convoluindo a imagem pela mascara.
    for i in range(k):
        for j in range(l):
            Bx[m:(h-m),n:(w-n)] += mask_sx[i,j] * A[i:(h-(k-i)+1), j:(w-(l-j)+1)]
            By[m:(h-m),n:(w-n)] += mask_sy[i,j] * A[i:(h-(k-i)+1), j:(w-(l-j)+1)]

    # Benchmarking
    # t2 = time.perf_counter()
    # print('sobel time elapsed: {} s'.format(t2-t1))
    
    # Retornar normalizado ou não
    if normalizado:
        return convert_pb_para_bin(normalize_uint8(np.sqrt(np.power(Bx,2) + np.power(By,2))),threshold)

    else:
        return np.sqrt(np.power(Bx,2) + np.power(By,2))


def sobel_slow(A: np.ndarray):
    '''
    EM DESUSO.

    Sobel utilizando muitas iterações

    A: imagem em e com 1 canal ndarray
    '''

    # Benchmarking
    # t1 = time.perf_counter()
    
    # Alocar espaço para as matrizes.
    B = np.zeros(A.shape)
    Gx = np.zeros(A.shape)
    Gy = np.zeros(A.shape)

    # Kernels de Sobel
    sy = np.array([[ -1,-2,-1], 
                   [  0, 0, 0],
                   [  1, 2, 1]])

    sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    # Dimensões do kernel
    m = sx.shape[0]//2; k = sx.shape[0]
    n = sx.shape[1]//2; l = sx.shape[1]
    # Dimensões da imagem
    w = A.shape[1]
    h = A.shape[0]

    for i in range (m, h-m):
        for j in range(n, w-n):
            Gx[i,j] = np.sum(np.multiply(sx , A[(i-m):(i+m+1), (j-n):(j+n+1)]))
            Gy[i,j] = np.sum(np.multiply(sy , A[(i-m):(i+m+1), (j-n):(j+n+1)]))            
    
    # Benchmarking
    # t2 = time.perf_counter()
    # print('sobel2: time elapsed: {} s'.format(t2-t1))

    # Retornar convertendo para bin aplicando threshold de 127
    return convert_pb_para_bin( normalize_uint8(np.sqrt(np.power(Gx,2) + np.power(Gy,2))), 127)


def derivativo_slow(A: np.ndarray):
    '''
    EM DESUSO:

    derivativo muitas iterações

    Retorna: Gx e Gy
    '''

    t1 = time.perf_counter()
    
    mask_x = np.array([[ -1,0,1]])
    mask_y = np.array([[-1],[0],[1]])

    w = A.shape[1]
    h = A.shape[0]

    # Fazer para a direção horizontal:
    Gx = np.zeros(A.shape)

    mask = mask_x
    m = mask.shape[0]//2
    n = mask.shape[1]//2
    k = mask.shape[0]
    l = mask.shape[1]

    for i in range (m, A.shape[0]-m):
        for j in range(n, A.shape[1]-n):
            Gx[i,j] = np.sum(np.multiply(mask_x , A[(i-m):(i+m+1), (j-n):(j+n+1)]))

    # Fazer para a direção vertical:
    Gy = np.zeros(A.shape)
    
    mask = mask_y
    m = mask.shape[0]//2
    n = mask.shape[1]//2
    k = mask.shape[0]
    l = mask.shape[1]
    for i in range (m, A.shape[0]-m):
        for j in range(n, A.shape[1]-n):
            Gy[i,j] = np.sum(np.multiply(mask_y , A[(i-m):(i+m+1), (j-n):(j+n+1)]))
    
    t2 = time.perf_counter()
    print('derivativo2: time elapsed: {} s'.format(t2-t1))
    
    return [normalize_uint8(Gx), normalize_uint8(Gy)]


def derivativo(A: np.ndarray, apply_threshold = False, threshold=127):
    '''
    derivativo poucas iterações
    
    Aplica mascara derivativa na direção dada.
    Retorna np.ndarray resultado.

    '''
    t1 = time.perf_counter()

    if len(A.shape) == 3:
        A = convert_color_para_pb(A)
    elif len(A.shape)==2:
        pass
    else:
        return np.zeros(A.shape)

    mask_x = np.array([[ -1,0,1]])
    mask_y = np.array([[-1],[0],[1]])

    # Fazer para a direção horizontal:
    Gx = np.zeros(A.shape)
    Gy = np.zeros(A.shape)
    
    # Para x
    mask = mask_x
    
    # Dimensões da mascara
    m = mask.shape[0]//2; k = mask.shape[0]
    n = mask.shape[1]//2; l = mask.shape[1]
    # Dimensões da imagem
    w = A.shape[1]
    h = A.shape[0]

    # Aplicar 
    for i in range(k):
        for j in range(l):
            Gx[m:(h-m),n:(w-n)] += mask[i,j] * A[i:(h-(k-i)+1), j:(w-(l-j)+1)]

    # Para y:
    mask = mask_y
    
    # Dimensões da mascara
    m = mask.shape[0]//2; k = mask.shape[0]
    n = mask.shape[1]//2; l = mask.shape[1]
    # Dimensões da imagem
    w = A.shape[1]
    h = A.shape[0]

    # Aplicar 
    for i in range(k):
        for j in range(l):
            Gy[m:(h-m),n:(w-n)] += mask[i,j] * A[i:(h-(k-i)+1), j:(w-(l-j)+1)]

    # Benchmarking
    # t2 = time.perf_counter() 
    # print('derivativo: time elapsed: {} s'.format(t2-t1))
    return np.sqrt(np.power(Gx,2)+np.power(Gy,2))


def aplica_mascara(img:np.ndarray, mask: np.ndarray, fast=True, normalize=True):
    '''
    img - Imagem a ser aplicada a máscara (apenas 1 canal de 8 bits)
    mask - mascara
    fast - método rápido (True), devagar(False)
    normalize - normalizar para 8 bits (True), senão retorna em float.
    
    retorna: Imagem normalizada para uint8 ou em float mesmo.

    '''
    imgRes = np.zeros(img.shape)

    # Parâmetros do tamanho da mascara
    m = mask.shape[0]//2 ; k = mask.shape[0]
    n = mask.shape[1]//2 ; l = mask.shape[1]

    # Parâmetros do tamanho da imagem
    w = img.shape[1]
    h = img.shape[0]

    # Método rápido: n iterações é o n de elementos na mascara
    if fast:
        for i in range(k):
            for j in range(l):
                imgRes[m:(h-m),n:(w-n)] += mask[i,j] * img[i:(h-(k-i)+1), j:(w-(l-j)+1)]
    
    # Método lento: n iterações é o n de elementos na imagem (menos os pixels de borda)
    else:
        for i in range (m, h-m):
            for j in range(n, w-n):
                imgRes[i,j] = np.sum(np.multiply(mask , img[(i-m):(i+m+1), (j-n):(j+n+1)]))

    if normalize:
        return normalize_uint8(imgRes)
    else:
        return imgRes


def kirsch(A:np.ndarray):
    '''
    A - IMAGEM
    retorna - imagem normalizada.
    '''
    # Alocar o lugar da imagem resultante
    R = np.zeros(A.shape)

    masks = [(1/15)* np.array([[-3,-3, 5], [-3, 0, 5], [-3,-3, 5]]), 
             (1/15)* np.array([[-3, 5, 5], [-3, 0, 5], [-3,-3,-3]]),
             (1/15)* np.array([[ 5, 5, 5], [-3, 0,-3], [-3,-3,-3]]),
             (1/15)* np.array([[ 5, 5,-3], [ 5, 0,-3], [-3,-3,-3]]),
             (1/15)* np.array([[ 5,-3,-3], [ 5, 0,-3], [ 5,-3,-3]]),
             (1/15)* np.array([[-3,-3,-3], [ 5, 0,-3], [ 5, 5,-3]]),
             (1/15)* np.array([[-3,-3,-3], [-3, 0,-3], [ 5, 5, 5]]),
             (1/15)* np.array([[-3,-3,-3], [-3, 0, 5], [-3, 5, 5]]) ]
    # Aplicar cada uma das mascaras e comparar guardando os maiores valores.
    for mask in masks:
        G = aplica_mascara(A, mask, normalize=False)
        R[G>R] = G[G>R] 
    
    return normalize_uint8(R)


########################## FIXME

def canny(img:np.ndarray, thresh=127, sigma=1):
    pass

###############################

def histograma(img:np.ndarray, normalize=True):
    '''
    calcular histogramas para 256 tons de cinza.
    img: np.ndarray em tons de cinza.
    retorna histograma np.ndarray (256,)
    '''

    # Check dimension:
    if len(img.shape) == 3:
        img = convert_color_para_pb(img)
    
    # Check data type
    if img.dtype != np.uint8(1).dtype:
        img = np.uint8(img)

    # Alocar espaço
    hist = np.zeros(256)
    for i in range(256):
        hist[i] = np.sum(img == i)

    # normalizar 
    if normalize:
        hist = hist / np.sum(hist)

    return hist


def cdf(hist:np.ndarray):
    '''
    Calcular a CDF de um histograma normalizado.
    hist em vetor de shape(n,)
    return cdf mesmo shape do vetor.
    '''
    # Alocar
    c = np.zeros(hist.shape)
    # Acumular
    for i in range(1, len(hist)):
        c[i] = c[i-1] + hist[i-1]
    
    return c


def equalizacao_histogramas(img: np.ndarray):
    '''
    
    '''
    # calcular a cdf do histograma
    f=cdf(histograma(img))

    # Alocar espaço
    img_res = np.zeros(img.shape, dtype= 'uint8')
    # Catar na função de probabilidade e multiplicar.
    # g[i, j] = img[i, j] * f(img[i, j])
    img_res = np.uint8(np.multiply(np.take(f, img), img))
    
    return img_res


def transformacao_intensidade(img: np.ndarray, 
                                f =31.875 * np.log2(np.arange(256) + 1)):
    '''
    img é a img
    f é o vetor de lookup ( f(range(256)) )
    retorna o f resultante
    '''

    # Alocar espaço
    img_res = np.zeros(img.shape, dtype= 'uint8')
    # Catar na função de probabilidade e multiplicar.
    # g[i, j] = img[i, j] * f(img[i, j])
    img_res = np.uint8(np.take(f, img))

    return img_res


def autoescala(img:np.ndarray):
    
    # Encontrar o máx.
    mx = np.max(img)
    # Encontrar o min
    mn = np.min(img)
    # Nova matriz normalizada.
    # Checar os valores de min e max. Se forem iguais, geram erros.
    if mn != mx:
        return np.uint8((255/ (mx-mn)) * (img - mn))
    else:
        return np.uint8(img)
    

def multi_limiarizacao(img, threshs= [127], values= [0, 255]):
    '''
    limiariza a imagem de acordo com os limiares em thresh(lista) ou iteravel
    thresh: iteravel contendo n limiares (em ordem cresecente)
    values: n+1 valores dos limiares contendo 
    
    Ex.: img_limiarizada = multi_limiarizaca(img, threshs=[100, 150], values=[20, 100, 255])
    
    retorna a img limiarizada.
    '''
    # Alocar
    img_lim = np.zeros(img.shape)

    t = values[-1] * np.ones(256, dtype='uint8')

    last= 0
    for trsh, val in zip(threshs, values):
        t[last:trsh] = val
        last = trsh

    return np.uint8(np.take(t, img))


def limiarizacao_global(img:np.ndarray, thresh=0):
    '''
    img: imagem em tons de cinza
    thresh: limiar inicial
            para média, deixe 0.
            para usar o seu, entre com inteiro
    retorna a img limiarizada e o threshold
    '''

    # Chute inicial para o threshold
    if thresh == 0:
        thresh = img.mean()

    # dT inicial
    dT = 255
    # limitar o numero de iterações
    i=0; iterlimit = 100

    while dT>2 and i< iterlimit:
        # print(thresh)
        M1 = (img[img< thresh]).mean()
        M2 = (img[img>=thresh]).mean()
        novo_thresh = 0.5* (M1+M2)
        dT = np.abs(thresh - novo_thresh)
        thresh = novo_thresh
        i+=1

    return (np.uint8(255*(img>=thresh)) , thresh)


def media_k(h, k):
    '''
    intensidade média até o nivel k de um histograma 256.
    '''
    return np.sum(np.multiply(np.arange(256), h)[0:int(k+1)])


def segmentacao_global_otsu(img:np.ndarray):
    '''
    encontra um limiar ótimo conforme o método de otsu
    retorna uma tupla com a imagem limiarizada e o limiar.
    '''
    # Calcular as propabilidades da imagem
    p = histograma(img) # P robabilidade (histograma)
    P = cdf(p)          # Cumulativa
    #P = P+0.00001
    mg = media_k(P, 255)   # media geral

    # alocar espaço
    var_k_2 = 0 * np.arange(256,dtype='float32')
    
    media_k_vec = np.multiply(np.arange(256), p)

    for k in range(256):
        var_k_2[k] = P[k] * (np.sum(media_k_vec[0:(k+1)]) - mg)**2 + \
                     (1-P[k]) * (np.sum(media_k_vec[k+1:]) - mg)**2
    
    k_max = np.argmax(var_k_2)
    
    #print(k_max)
    # plt.subplot(2,2,1)
    # plt.bar(np.arange(256), p, label= 'p'); plt.legend()
    # plt.subplot(2,2,2)
    # plt.plot(P, label='P'); plt.legend()
    # plt.subplot(2,2,3)
    # plt.plot(var_k, label='var_k'); plt.legend()
    # plt.subplot(2,2,4)
    # plt.plot(var_k_2, label='var_k_2'); plt.legend()
    # plt.show()

    return (np.uint8(255*(img>= int(k_max))) ,int(k_max))


def aplica_mascara_2(img:np.ndarray, mask: np.ndarray):
    '''
    img - Imagem a ser aplicada a máscara (apenas 1 canal de 8 bits)
    mask - mascara

    => nessa implementação, não são deixadas bordas com valor nulo. a imagem é expandida
        com zeros e filtrada, de modo que os pixels das bordas são resultados da filtragem
         também.
    
    retorna: imagem resultado da convolução da mascara. sem normalização.

    '''

    # Parâmetros do tamanho da mascara
    m = mask.shape[0]//2 ; k = mask.shape[0]
    n = mask.shape[1]//2 ; l = mask.shape[1]

    # Parâmetros do tamanho da imagem
    w = img.shape[1]
    h = img.shape[0]

    # reshape image:

    # Alocar o espaço para nova imagem
    imgRes = np.zeros((h+2*m, w+2*n))

    imgOrgAdj = np.zeros((h+2*m, w+2*n))

    imgOrgAdj[m:(m+img.shape[0]), n:(img.shape[1]+n)] = img[:, :]

    # Parâmetros do tamanho da imagem
    w = imgOrgAdj.shape[1]
    h = imgOrgAdj.shape[0]

    # Método rápido: n iterações é o n de elementos na mascara
    for i in range(k):
        for j in range(l):
            imgRes[m:(h-m),n:(w-n)] += mask[i,j] * imgOrgAdj[i:(h-(k-i)+1), j:(w-(l-j)+1)]

    return imgRes[m:(img.shape[0]+m), n:(img.shape[1]+n)]


class masks():
    def __init__(self):
        pass
    
    @staticmethod
    def media(n):
        '''
        retorna uma mascara de média (n,n)
        '''
        return (1/(n*n))*np.ones((n,n))

    @staticmethod
    def gaussiano(n, sigma=1):
        '''
        retorna uma mascara gaussiana (n,n) de desvio sigma.
        '''
        size = int(n) // 2
        x, y = np.meshgrid(np.arange(-size,size+1), np.arange(-size,size+1))
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    @staticmethod
    def laplaciano(n=5):
        '''
        n = 3, 5 ou  9
        se nada, retorna o mesmo para n=5
        '''

        if n== 3:
            return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        elif n==5:
            m = -1*np.ones((5,5))
            m[2,2] = 24
            return m
        
        elif n==9:
            m = -1*np.ones((9,9))
            m[3:6,3:6] = 8
            return m
        else:
            m = -1*np.ones((5,5))
            m[2,2] = 24
            return m

    @staticmethod
    def passa_altas(n=5):
        '''
        retorna uma mascara passa altas (n,n)
        '''
        if n%2 == 0:
            n = n + 1
        elif n<3:
            n = 3
        
        mascara = (-1) * np.ones((n,n))
        mascara[n//2, n//2] = (-1) * np.sum(mascara) - 1

        return mascara


def adicionar_ruido(img:np.ndarray, tipo= 'uniforme', quantidade= 0.15, amplitude = 30):
    '''
    adiciona ruido na imagem.
    img = img + amplitude * ruido
    quantidade: percentual de pontos da imagem que serão afetados pelo ruido
    (em decimal)
    tipo: 'uniforme' ou 'gaussiano'

    '''
    # Com auxilio da distribuição uniforme vamos fazer com que apenas uma parte
    # seja diferente de zero.
    ruido = 1 * (np.random.rand(img.shape[0],img.shape[1]) < quantidade)

    # Caso seja uniforme rand nos dá numeros uniformemente distribuidos
    if tipo == 'uniforme':
        ruido = np.multiply(ruido, 2* np.random.rand(img.shape[0],img.shape[1]) - 1)
    
    # Caso seja gaussiano randn nos dá numeros distribuidos normalmente
    elif tipo == 'gaussiano':
        ruido = np.multiply(ruido,    np.random.randn(img.shape[0],img.shape[1]) )
    
    # Por default retorna uniforme
    else:
        #  2*rand -1 força os números aleatórios a saírem entre -1 e 1
        ruido = np.multiply(ruido, 2* np.random.rand(img.shape[0],img.shape[1])-1)
    
    # Força todos os numeros a ficarem dentro de 0 a 255, sem normalizá-los.
    return clip_uint8(img + amplitude*ruido)


def pseudomediana(A:np.ndarray):
    '''
    Encontra a pseudomediana para um vetor
    '''
    
    L = A.size
    M = (L+1)//2

    mmx_counter = 0
    mins = np.zeros(M)
    maxs = np.zeros(M)

    for p in range(L-M+1):
        mmx_array = A[p:(p+M)]
        mins[mmx_counter] = np.min(mmx_array)
        maxs[mmx_counter] = np.max(mmx_array)
        mmx_counter+=1

    maxmin = np.max(mins)
    minmax = np.min(maxs)

    pmediana = int((maxmin+minmax)//2)

    return pmediana


def filtro_pseudomediana(img:np.ndarray, tamanho= 3):

    # Expandir as bordas para aplicar a mascara às bordas tb.

    # Parâmetros do tamanho da mascara
    m = tamanho//2 ; k = tamanho
    n = tamanho//2 ; l = tamanho

    # Parâmetros do tamanho da imagem
    w = img.shape[1]
    h = img.shape[0]

    # reshape image:

    # Alocar o espaço para nova imagem (Expandir a borda para que o centro da 
    # mascara comece no primeiro elemento da imagem)
    imgRes = np.zeros((h+2*m, w+2*n))
    imgAdj = np.zeros((h+2*m, w+2*n))
    imgAdj[m:(m+img.shape[0]), n:(img.shape[1]+n)] = img[:, :]

    # Parâmetros do tamanho da imagem nova imagem ajustada
    w = imgAdj.shape[1]
    h = imgAdj.shape[0]

    # Aplicar a pseudomediana:
    '''
    PMED = MAXMIN(S_L) + MINMAX(S_L)
    '''
    L = tamanho**tamanho
    M = (tamanho+1)//2

    # Loop por todos os elementos da imagem ajustada
    for i in range(m, h-m):
        for j in range(n, w-n):
            # encontrar a pseudomediana e atribuir a i, j na imagem resultante
            imgRes[i,j] = pseudomediana(
                    imgAdj[(i-m):(i+m+1), (j-m):(j+n+1)].flatten())

        
    return imgRes[m:(img.shape[0]+m), n:(img.shape[1]+n)]


def dilatar(img:np.ndarray,element:np.ndarray, center= (0,0), limiar = 0):
    '''
    Dilatação de elementos maiores que limiar.
    retorna matriz de booleans
    '''
    # Converter para binarios (booleanos)
    img_bool = img>limiar
    element_bool = element>limiar

    # Parâmetros do tamanho da mascara
    m = element.shape[0]//2 ; k = element.shape[0]
    n = element.shape[1]//2 ; l = element.shape[1]

    # Parâmetros do tamanho da imagem
    w = img.shape[1]
    h = img.shape[0]

    # reshape image:

    # Alocar o espaço para nova imagem
    imgRes = np.zeros((h+2*m, w+2*n))

    imgOrgAdj = np.zeros((h+2*m, w+2*n))

    imgOrgAdj[m:(m+img.shape[0]), n:(img.shape[1]+n)] = img[:, :]

    # Parâmetros do tamanho da imagem
    w = imgOrgAdj.shape[1]
    h = imgOrgAdj.shape[0]

    #FIXME: Problema quando o elemento tem alguma dimensão par
    for i in range (m, h-m):
        for j in range(n, w-n):
            # Se tiver um pixel marcado na imagem (descontando o centro da mascar
            # que pode ser escohido pelo usuário)
            if (imgOrgAdj[i-center[0],j-center[1]]):
                imgRes[(i-m):(i-m+k), (j-n):(j-n+l)] = \
                    np.logical_or(
                        imgRes[(i-m):(i-m+k), (j-n):(j-n+l)], 
                        element)


    return imgRes[m:(img.shape[0]+m), n:(img.shape[1]+n)]


def erodir(img:np.ndarray,element:np.ndarray, center= (0,0), limiar = 0):
    '''
    Erosão de elementos maiores que limiar.
    retorna matriz de booleans
    '''
    # Converter para binarios (booleanos)
    img_bool = img>limiar
    element_bool = element>limiar

        # Parâmetros do tamanho da mascara
    m = element.shape[0]//2 ; k = element.shape[0]
    n = element.shape[1]//2 ; l = element.shape[1]

    # Parâmetros do tamanho da imagem
    w = img.shape[1]
    h = img.shape[0]

    # reshape image:

    # Alocar o espaço para nova imagem
    imgRes = np.zeros((h+2*m, w+2*n))

    imgOrgAdj = np.zeros((h+2*m, w+2*n))

    imgOrgAdj[m:(m+img.shape[0]), n:(img.shape[1]+n)] = img[:, :]

    # Parâmetros do tamanho da imagem
    w = imgOrgAdj.shape[1]
    h = imgOrgAdj.shape[0]

    #FIXME: Problema quando o elemento tem alguma dimensão par
    for i in range (m, h-m):
        for j in range(n, w-n):
            imgRes[i-m+center[0],j-n+center[1]] = \
                np.all(np.logical_and(imgOrgAdj[(i-m):(i-m+k), (j-n):(j-n+l)], element))


    return imgRes[m:(img.shape[0]+m), n:(img.shape[1]+n)]


def abertura(img:np.ndarray, elem:np.ndarray, centro= (0,0), limiar=0):
    imgRes = 255*erodir(img, 1 * np.logical_not(elem), centro, limiar=limiar)
    imgRes = dilatar(img, elem, centro, limiar=limiar)
    
    return imgRes


def fechamento(img:np.ndarray, elem:np.ndarray, centro= (0,0), limiar=0):
    imgRes = 255*dilatar(img, 1 * np.logical_not(elem), centro, limiar=limiar)
    imgRes = erodir(img, elem, centro, limiar=limiar)
    
    return imgRes


def segmentacao_a_la_eduardo(img_in:np.ndarray, limiar=0):
    '''
    recebe a imagem limiarizada

    retorna: imagem segmentada, numero de elementos
    '''
    # Expandir a imagem para eliminar os problemas de indices em bordas
    img = np.zeros((img_in.shape[0]+2, img_in.shape[1]+2))
    imgRes = np.zeros(img.shape,dtype='int')
    # Colocar a img no espaço alocado
    img[1:-1, 1:-1] = img_in

    # inicalizar os contadores de regiões
    region_counter = 0
    current_counter = 0

    # Escanear elemento a elemento:
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            
            # Se for um pixel marcado:
            if img[i, j]:
                
                # Verificar se já há marcação adjacente
                if (imgRes[i-1,j] or imgRes[i, j-1]):
                
                    # se for o pixel de cima que estiver marcado:
                    if imgRes[i-1,j]:
                        current_counter = imgRes[i-1,j]

                    # se o pixel anterior que estiver marcado:
                    elif imgRes[i, j-1]:
                        current_counter = imgRes[i,j-1]
                    
                    # Marcar o que tiver que marcar
                    imgRes[i-1, j] = current_counter if (imgRes[i-1, j]) else imgRes[i-1, j]
                    imgRes[i, j-1] = current_counter if (imgRes[i, j-1]) else imgRes[i, j-1]
                    imgRes[i,j]    = current_counter

                else:
                    # Incrementaro o contador de regiões e assumir que é uma região nova
                    region_counter += 1
                    current_counter = region_counter

                    # Marcar como uma nova região
                    imgRes[i,j] = current_counter
                    
                    # Marcar o que tiver que marcar à direita e abaixo
                    imgRes[i+1, j] = current_counter if img[i+1, j] else imgRes[i+1, j]
                    imgRes[i, j+1] = current_counter if img[i, j+1] else imgRes[i, j+1]
    #######

    # Verificar por regiões onde há conflitos de regiões:
    # Escanear elemento a elemento:
    for i in range(img.shape[0]-1, 0, -1 ):
        for j in range(img.shape[1]-1, 0, -1):
            # Caso seja marcado
            if imgRes[i, j]:

                # Se o superior estiver ativo e for diferente 
                if imgRes[i-1, j] and imgRes[i-1, j]!= imgRes[i, j]:
                    imgRes[i-1, j] = imgRes[i, j]

                # Se o anterior estiver ativo e for diferente 
                if imgRes[i, j-1] and imgRes[i, j-1]!= imgRes[i, j]:
                    imgRes[i, j-1] = imgRes[i, j]
    ###
    # Reorganizar as regiões
    vals = list(range(1,region_counter+1)) #[1 2 3 4 5 6 7 8 9 10]
    h    = list(np.zeros(len(vals)))        

    #
    n_regions = len(vals)
    

    # criar um histograma
    for i, val in enumerate(vals):
        h[i] = np.sum(imgRes == val)      #[1 2 3 x x x 7 8 x 10]

    # Caso haja algum zero: corrigir
    if not np.all(h):
        for i in range(len(vals)):
            if h[i] == 0:
                h[i]    =-1
                vals[i] =-1
        
        # Excluir os zeros
        while(True):
            try:
                h.remove(-1)
            except:
                break

        while(True):
            try:
                vals.remove(-1)
            except:
                break    
        
        #refazer o histograma:
        vals_novos = np.arange(1,len(vals)+1) #[1 2 3 4 5 6 ]
        
        for val_antigo, val_novo in zip(vals, vals_novos):
            imgRes[imgRes== val_antigo] = val_novo

        n_regions = vals_novos[-1]

    return imgRes[1:-1, 1:-1], n_regions


def posicao(img:np.ndarray, value= 1):
    '''
    retorna a posição e a área de um objeto com valor == value
    na img segmentada.

    return x, y, area
    '''
    # Meshgrid para facilitar no calculo das médias.
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    # print('xx')
    # print(xx)
    # print('yy')
    # print(yy)

    # Calcular a área do obj
    ar = calcular_area(img, 1)
    if ar==0:
        ar = 1
    # Encontrar onde ele esta na img
    cond = img == value
    # Calcular a média horizontal
    x_ = int(np.round( np.sum(xx[cond]) / ar))
    # Calcular a media vertical
    y_ = int(np.round( np.sum(yy[cond]) / ar))

    return x_, y_, ar


def orientacao(img:np.ndarray, value= 1):
    '''
    retorna a posição, orientação em graus e a área de um objeto com valor == value
    na img segmentada.

    return x, y, theta, area
    '''

    # Meshgrid para facilitar no calculo das médias.
    xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    # Calcular a área do obj
    ar = calcular_area(img, value)
    # Encontrar onde ele esta na img
    cond = img == value
    # Calcular a média horizontal
    x_ = int(np.round( np.sum(xx[cond]) / ar))
    # Calcular a media vertical
    y_ = int(np.round( np.sum(yy[cond]) / ar))


    #
    x_linha = np.multiply((xx - x_), 1 * cond)
    y_linha = np.multiply((yy - y_), 1 * cond)

    # a
    a = np.sum(x_linha**2)

    # b
    b = 2 * np.sum( np.multiply(x_linha, y_linha) )

    # c
    c = np.sum(y_linha**2)

    # Calcular o ângulo em rad.
    theta = - np.arctan2(b , (a-c)) / 2

    # Converter para graus:
    theta = np.round(180 * theta / np.pi, decimals = 1)
    #print('theta = ', 180 * theta / np.pi)

    return x_, y_, theta, ar 


def calcular_area(img:np.ndarray, value = 1):
    return np.sum(img == value)


def comprimento_e_largura(img_in:np.ndarray, value):
    '''
    Realiza a extração de características de uma imagem segmentada.

    retorna a posição, orientação em graus e a área de um objeto com valor == value
    na img segmentada.

    TODO: Atualizar lista de saidas
    '''
    # Calcular a área do obj
    ar = calcular_area(img_in, value)

    # Meshgrid para facilitar no calculo das médias.
    xx, yy = np.meshgrid(range(img_in.shape[1]), range(img_in.shape[0]))

    # Encontrar onde ele esta na img. cond contém a info. lógica de quais pixels dessa img são
    # do objeto. será bastante util durante esse algoritmo para controlar bugs.
    cond = img_in==value

    # filtrar o meshgrid
    xs = xx[cond]
    ys = yy[cond]

    # Encontrar a janela quardada em que o objeto se localiza
    x_max = np.max(xs); x_min = np.min(xs)
    y_max = np.max(ys); y_min = np.min(ys)
    
    # Calcular a largura da janela quadrada do obj. 
    dx = x_max - x_min + 1
    dy = y_max - y_min + 1 
    

    # Calcular a média horizontal
    x_ = int(np.round( np.sum(xs) / ar))
    # Calcular a media vertical
    y_ = int(np.round( np.sum(ys) / ar))

    # Diferença entre cada ponto pertencente ao obj ao centro da img
    x_linha = np.multiply((xx - x_), 1 * cond)
    y_linha = np.multiply((yy - y_), 1 * cond)

    # a
    a = np.sum(x_linha**2)

    # b
    b = 2 * np.sum( np.multiply(x_linha, y_linha) )

    # c
    c = np.sum(y_linha**2)

    # Calcular o ângulo de orientação em rad.
    theta = - np.arctan2(b , (a-c)) / 2

    # Calcular utilizando as propriedades de projeção de vetores. Projetar um vetor(x - x_medio, y- y_medio)
    # na direção da orientação do elemento
    proj_comprimento = x_linha * np.cos(theta) + y_linha*np.sin(theta)
    proj_largura     = x_linha * (- np.sin(theta)) + y_linha * np.cos(theta)

    # theta para graus:
    theta = 180 * theta / np.pi

    # Calcular as projeções
    h = int(np.round(np.max(proj_comprimento) - np.min(proj_comprimento) + 1))
    w = int(np.round(np.max(proj_largura    ) - np.min(proj_largura    ) + 1))

    return x_, y_, theta, w, h, ar, dx, dy, (x_min, y_min), (x_max, y_max)


def limiarizacao_por_bordas(img: np.ndarray, limiar_bordas = 127):
    '''
    Retorna um limiar obtido pela média dos pixels de borda 
    acima de 127 (sendo os pixels da borda normalizados entre 0 e 255)

    img possui apenas um canal.
    '''
    # Extrair as bordas da img usando sobel.
    bordas = sobel(img, normalizado= True) # entre 0 e 255

    # Thresh = média dos pixels selecionados de borda
    thresh = np.sum(img[bordas>limiar_bordas]) / np.sum(bordas>limiar_bordas)
    
    return (np.uint8(255*(img>=thresh)) , thresh)


