'''
Projeto final de visão computacional 2020-1
Aluno: Renan Praciano Ideburque Leal Sandes
Professor: Eduardo O. Freire
Data: Agosto 2020


'''

import numpy as np
import cv2
from operacoes_viscomp import *
import matplotlib.pyplot as plt

# Print init:
# Instruções para uso do programa
print('Iniciando a captura...')
print('Press q para sair, ou quando concluir a calibração.')
print('Pressione v para alternar a visualização das infos')

### Def variáveis ++++++++++++++++++++++++++++

# Inicializar o dispositivo
cam = cv2.VideoCapture(0)

# Constantes range da cor da pele in YCrCb
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

# Valores para a região da captura: para serem usados na imagem 480, 640, 3
# Definir dicionários que vão ajudar a conciliar as regiões de interesse e o 
# desenho dos retângulos.
x, y = 28, 28
l, a = 300, 300
s = 3

hand_region   = {'p1': (x, y), 'p2': (x+l, y+a), 'color': (  0,   0, 255), 'img': np.array([0])}
palm_cut      = {'p1': (x, y), 'p2': (x+l, y+a), 'color': (255,   0, 255), 'img': np.array([0])}
finger_region = {'p1': (x, y), 'p2': (x+l, y+a), 'color': (  0, 255, 255), 'img': np.array([0])}

# Valores para a img 160, 120 | escala de redução = 1/4
x, y = 7, 7
l, a = 75, 75
s = 3

# Imagem da mão toda: 75x75 (Pedaço da imagem com dimensão reduzida (160,120) )
hand_region_s   = {'p1': (x, y), 'p2': (x+l, y+a), 'color': (  0,   0, 255), 'img': np.zeros((120,160), dtype= np.uint8)}
# Imagem com a palma cortada (Mesma dim de hand_region_s: (160,120))
palm_cut_s      = {'p1': (x, y), 'p2': (x+l, y+a), 'color': (255,   0, 255), 'img': np.zeros((120,160), dtype= np.uint8)}
# Pedaço da imagem contendo os dedos (Mesma dim de hand_region_s: (160,120))
finger_region_s = {'p1': (x, y), 'p2': (x+l, y+a), 'color': (  0, 255, 255), 'img': np.zeros((120,160), dtype= np.uint8)}
# Pedaço da imagem contendo o polegar (Mesma dim de hand_region_s: (160,120))
thumb_region_s  = {'p1': (x, y), 'p2': (x+l, y+a), 'color': (  0, 255, 255), 'img': np.zeros((120,160), dtype= np.uint8)}

parametro = 15


### Def funções ++++++++++++++++++++++++++++++
# Quebrar o algoritmo em funções para facilitar a leitura do loop.
# Para ler mais rápido, pule para a seção do Loop Principal
def capturar(cam):
    ''' capturar imagem da camera '''
    # Retorna a img capturada pela camera
    _, img = cam.read()
    return img

def pre_processar(img):
    ''' pre-processar antes de extrair filtrando por cor'''
    ### Redimensionar a imagem para agilizar o processo:
    img_s = cv2.resize(img, (160,120), cv2.INTER_AREA)
    
    ### Aplicar filtro de média
    m = 3
    # img_media_s = img_s.copy() # Só pra preservar a img reduzida inicial
    # img_media_s[:,:,0] = aplica_mascara_2(img_s[:,:,0], masks.media(m))
    # img_media_s[:,:,1] = aplica_mascara_2(img_s[:,:,1], masks.media(m))
    # img_media_s[:,:,2] = aplica_mascara_2(img_s[:,:,2], masks.media(m))
    img_pre_processada = cv2.blur(img_s, (m, m))

    return img_pre_processada

def extrair_cor_pele(img: np.ndarray, min_YCrCb=np.array([0,133,77],np.uint8) , max_YCrCb= np.array([255,173,127],np.uint8)):
    ''' Retornar uma img binaria onde são as cores da mão '''

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    
    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    return skinRegion 

def extrair_regiao_interesse(img_escala_reduzida, hand_region_s: dict):
    ''' Extrair as regiões de interesse: '''

    hand_region_s['img'] = img_escala_reduzida[hand_region_s['p1'][1]: hand_region_s['p2'][1], hand_region_s['p1'][0]: hand_region_s['p2'][0]]
    return hand_region_s

def recorte_da_palma_da_mao(hand_region_s: dict, palm_cut_s: dict):
    ''' encontrar a palma e retornar o recorte da palma'''
    
    # Encontrar a palma utilizando erosão:
    tamanho_mascara_erosao = 21; centro_mascara_erosao = (10,10)
    palm = 1 * erodir(hand_region_s['img'][:, :], np.ones((tamanho_mascara_erosao,tamanho_mascara_erosao)), center= centro_mascara_erosao)
    x_, y_, ar = posicao(palm, value=1)
    palm_position = (x_, y_)
    # print('area palma: ', ar)
    calibracao_proximidade_palma = 120
    palm_is_on = ar> calibracao_proximidade_palma # Condicional pra dizer se a palma está na img   

    # Mostrar a região da mão após a erosão para encontrar a palma da mão
    #mostrar('palm', palm, 4)

    # ROI: mão sem a palma:
    # palm_cut_s['img'] = hand_region_s['img'].copy()
    
    palm_cut_s['img'] = np.uint8(255*palm) # FIXME Caso dê errado

    # Valores para o corte da palma da mão: Valores referentes a hand_region_s
    #(if_test_is_false, if_test_is_true)[test]
    cut_size = 21; offset_x = 10; offset_y = 5

    y0 = (y_-cut_size+offset_y, 0 )[y_-cut_size+offset_y<0 ]
    x0 = (x_-cut_size+offset_x, 0 )[x_-cut_size+offset_x<0 ]
    y1 = (y_+cut_size+offset_y, 76)[y_+cut_size+offset_y>76]
    x1 = (x_+cut_size+offset_x, 76)[x_+cut_size+offset_x>76]
    
    palm_cut_s['p1'] = (x0, y0); palm_cut_s['p2'] = (x1, y1)
    
    return palm_is_on, palm_position

def encontrar_o_polegar(img_s, hand_region_s: dict, palm_cut_s: dict, thumb_region_s: dict):
    # Encontrar o polegar e 
    # thumb_region_s
    
    y0 = palm_cut_s['p1'][1]+ 15 #if palm_cut_s['p1'][1]>= 0 else 0
    y1 = palm_cut_s['p2'][1] #if palm_cut_s['p1'][1]>= 0 else 5

    x0 = palm_cut_s['p2'][0] + 5  #if palm_cut_s['p2'][0] + 5 >=0 else 0
    
    x0 = palm_cut_s['p2'][0] + 8 if palm_cut_s['p2'][0] + 5 > 6 else 70 
    x1 = palm_cut_s['p2'][0] + 15 if palm_cut_s['p2'][0] + 5 > 6 else 75 

    thumb_region_s['p1'] = (x0, y0); #print(thumb_region_s['p1'])
    thumb_region_s['p2'] = (x1, y1); #print(thumb_region_s['p2'])

    thumb_region_s['img'] = img_s[y0:y1, x0:x1]

    #print('shape ', thumb_region_s['img'].shape, 'p1: ', thumb_region_s['p1'],' p2: ', thumb_region_s['p2'])
    #mostrar('thumb', thumb_region_s['img'], 4)
    
    ar = np.sum(1* (thumb_region_s['img']>0))
    thumb = ar>10 #FIXME
    
    return thumb

def encontrar_os_dedos(hand_region_s: dict, palm_cut_s:dict, palm_is_on):
    
    if palm_is_on:
        # Ponto inicial da figura em x: 
        x0 = hand_region_s['p1'][0] # Daqui vou tirar o x0
        
        x0 = palm_cut_s['p1'][0] - 15 if (palm_cut_s['p1'][0] - 15 )>= 0 else 0 #FIXME
        # Ponto final da figura em x:
        x1 = hand_region_s['p2'][0] # daqui vou tirar o x1
        x1 = palm_cut_s['p2'][0] + 15 if (palm_cut_s['p2'][0] + 15) <= 75 else 75 # FIXME
        
        # Ponto inicial da figura em 3
        t = 11; b = 8

        y0 = palm_cut_s['p1'][1] - t if (palm_cut_s['p1'][1] - t)>= 5 else 5
        y1 = palm_cut_s['p1'][1] - b  if (palm_cut_s['p1'][1] - b)>= 0 else 10
        
        #
        finger_region_s['p1'] = (x0, y0); #print(finger_region_s['p1'])
        finger_region_s['p2'] = (x1, y1); #print(finger_region_s['p2']) 
        
        delta = 5 # Distancia do palm cut ao final dos dedos(eu acho) desconto para casar com o desenho do quadrado

        finger_region_s['img']= hand_region_s['img'][(y0-delta):(y1-delta), x0:(x1)]

        fingers_segm, n_fingers = segmentacao_a_la_eduardo(finger_region_s['img'], limiar = 0)
        
        #print(n_fingers, '  fingers_segm.shape = ', fingers_segm.shape, 'hand.shape: ', hand_region_s['img'].shape ,' hand p1: ', hand_region_s['p1'] ,' hand p2: ', hand_region_s['p2'] )

        finger_region_s['img'] = np.uint8(50*fingers_segm)
        
        print('--')
        k = 0
        for i in range(1, n_fingers+1):
            
            _, _, _, _, _, _, _, _, (x_min, y_min), (x_max, y_max) = comprimento_e_largura(fingers_segm, i)
            k = k + int(np.round((x_max - x_min)/10))
            #print(i, (np.ceil(x_max - x_min)//10))


        return k

def mostrar(nome, img, scale= 1):
    ''' facilitar a leitura e tb ajustar a escala para ampliar '''
    img_resized_to_show = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_AREA)
    cv2.imshow(nome, img_resized_to_show)

def sobrepor_info_na_img(img_cam, img_ex_s, palm_is_on, palm_position, n_fingers, thumb):
    ''' colocar todos os quadrados e textos na img pra ser exibida pro usuário '''

    # replicar de 1 para 3 canais:
    img_ex_s = cv2.cvtColor(img_ex_s, cv2.COLOR_GRAY2BGR)

    img_ex = img_cam # cv2.resize(img_ex_s, (640,480), cv2.INTER_AREA)
    
    # String esse -36 é para que a posição seja referenciada ao centro do quadrado
    x_ = palm_position[0]; y_ = palm_position[1]
    # n_fingers = 0
    # thumb = 0 

    if palm_is_on:
        # FIXME!
        #my_string = '({}, {}) | f= {} |t= {}'.format(int(x_-36), int(y_-36), n_fingers, thumb)
        my_string = '{}  {}'.format(n_fingers, thumb)

    else:
        my_string = '-'
    
    # Exibir a imagem re ampliada:de volta para 480x640
    cv2.putText(img_ex, my_string, (hand_region['p1'][0], hand_region['p1'][0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
    # Hand:
    cv2.rectangle(img_ex, (hand_region['p1'][0]-1, hand_region['p1'][1]-1), (hand_region['p2'][0]+1, hand_region['p2'][1]+1), hand_region['color'], 1)
    # Fingers:
    cv2.rectangle(img_ex, (finger_region['p1'][0]-1, finger_region['p1'][1]-1), (finger_region['p2'][0]+1, finger_region['p2'][1]+1), finger_region['color'], 1)
    # Palm:
    cv2.rectangle(img_ex, (palm_cut['p1'][0]-1, palm_cut['p1'][1]-1), (palm_cut['p2'][0]+1, palm_cut['p2'][1]+1), palm_cut['color'], 1)
       
    # Exibir a imagem pequena
    cv2.putText(img_ex_s, my_string, (hand_region_s['p1'][0], hand_region_s['p2'][1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)
    # Hand:
    cv2.rectangle(img_ex_s, (hand_region_s['p1'][0]-1, hand_region_s['p1'][1]-1), (hand_region_s['p2'][0]+1, hand_region_s['p2'][1]+1), hand_region_s['color'], 1)
   
    # Palm:
    if palm_is_on:
        # Palm:
        cv2.rectangle(img_ex_s, (palm_cut_s['p1'][0]-1, palm_cut_s['p1'][1]-1), (palm_cut_s['p2'][0]+1, palm_cut_s['p2'][1]+1), palm_cut_s['color'], 1)

        # Desenhar o recorte dos dedos:
        cv2.rectangle(img_ex_s, (finger_region_s['p1'][0]-1,finger_region_s['p1'][1]-1), (finger_region_s['p2'][0]+1, finger_region_s['p2'][1]+1), (255, 0, 255), 1)

        # Desenhar o recorte do polegar mão direita:
        cv2.rectangle(img_ex_s, (thumb_region_s['p1'][0]-1,thumb_region_s['p1'][1]-1), (thumb_region_s['p2'][0]+1, thumb_region_s['p2'][1]+1), thumb_region_s['color'], 1)

        # Fingers:
        cv2.rectangle(img_ex_s, (finger_region_s['p1'][0]-1, finger_region_s['p1'][1]-1), (finger_region_s['p2'][0]+1, finger_region_s['p2'][1]+1), finger_region_s['color'], 1)
    

        # # Desenhar o recorte do polegar mão direita:
        #cv2.rectangle(img_ex_s, (thumb_region_2_s['p1'][0]-1,thumb_region_2_s['p1'][1]-1), (thumb_region_2_s['p2'][0]+1, thumb_region_2_s['p2'][1]+1), thumb_region_2_s['color'], 1)


    return img_ex, img_ex_s


### Loop principal ==========================================

# Inicializar a variável de controle do loop. Quebra o loop ao pressionar tecla "q".
keyPressed = -1
userView = True

while keyPressed != ord('q'):
    # Capturar da camera
    captura = capturar(cam)
    
    # Pre processamento
    captura_pre_processada = pre_processar(captura)
    
    # Extrair cor da pele:
    pele = extrair_cor_pele(captura_pre_processada)

    # Extrair região da mão:
    hand_region_s = extrair_regiao_interesse(pele, hand_region_s)

    # Extrair palma da mão:
    palm_is_on, palm_position = recorte_da_palma_da_mao(hand_region_s, palm_cut_s)

    # Encontrar o polegar
    thumb = encontrar_o_polegar(pele, hand_region_s, palm_cut_s, thumb_region_s)

    # Econtrar os dedos
    n_fingers = encontrar_os_dedos(hand_region_s, palm_cut_s, palm_is_on)
    
    # Colocar as indicações na tela.
    img_para_exibicao, img_para_exibicao_s = sobrepor_info_na_img(captura, pele, palm_is_on, palm_position, n_fingers, thumb)
    
    if userView:
        mostrar('img_para_exibicao', img_para_exibicao, scale= 1)

    else:
        mostrar('img_para_exibicao_s', img_para_exibicao_s, scale= 4)
        #mostrar('palm_cut_s', palm_cut_s['img'], 2)
        mostrar('fingers_segm', finger_region_s['img'], scale = 5)

    #mostrar('hand_region_s', hand_region_s['img'], scale= 1)


    # ----------------------- 

    # Aguardar o teclado e esperar uns milisegundos
    keyPressed = cv2.waitKey(10)

    if keyPressed == ord('p'):
        parametro = parametro + 1
        print('parametro = ', parametro)

    if keyPressed == ord('o'):
        parametro = parametro - 1
        print('parametro = ', parametro)

    if keyPressed == ord('v'):
        userView = not userView
        cv2.destroyAllWindows()

    if keyPressed == ord('c'):
        # cv2.imwrite('imgs/00-export-cam.png', captura)
        # cv2.imwrite('imgs/01-export-pele.png', pele)
        # cv2.imwrite('imgs/02-export-hand_region_s.png', hand_region_s['img'])
        # cv2.imwrite('imgs/03-export-.png', finger_region_s['img'])
        # cv2.imwrite('imgs/04-export-.png', img_para_exibicao)
        # cv2.imwrite('imgs/05-export-.png', img_para_exibicao_s)
        # cv2.imwrite('imgs/06-export-.png', thumb_region_s['img'])
        # cv2.imwrite('imgs/07-export-.png', palm_cut_s['img'])
        pass


        


# Final do Loop principal do programa ===============================
 
# Liberar o dispositivo
cam.release()

# Terminar todas as janelas
cv2.destroyAllWindows()

