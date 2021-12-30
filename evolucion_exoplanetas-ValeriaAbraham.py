from matplotlib.pyplot import figure, plot, legend, show, xlabel, ylabel, title, ticklabel_format, subplots
from numpy import sin, cos, pi, arange, sqrt, array

###############################################################################
########################### Definición de funciones ###########################
###############################################################################

def evolucion(a,b,N,x,mp,mc,Rp,ac,Qp): 
    
    # Función que calcula la evolución orbital de un sistema planetario de dos planetas 
    # y una estrella, teniendo en cuenta el efecto de marea sobre la estrella y el 
    # planeta más cercano

    # a y b: extremos de integración
    # N: cantidad de pasos
    # x: vector con las condiciones iniciales
    # mp y mc: masas del planeta interno y externo respectivamente
    # Rp: radio del planeta interno
    # ac: semieje mayor del planeta externo
    # Qp: constante de disipación del planeta interno (afectado por marea)
    
    # Masa reducida del planeta principal y del compañero 
    mu = G * (ms + mp)
    muc = G * (ms + mc)
    
    nc = sqrt(muc/ac**3) # Movimiento medio del compañero
    
    s = 9/4 * kds/Qs * mp/ms * Rs**5 # Efecto de marea sobre la estrella
    p = 9/2 * kdp/Qp * ms/mp * Rp**5 # Efecto de marea sobre el planeta
    
    def rk4(a,b,N,x): # Rutina de Runge - Kutta 4
        h = (b-a)/N # Definimos el espaciado temporal 
        tp = arange(a,b,h) # Definimos el vector con todos los tiempos utilizados para luego graficar
        
        # Definimos las diferentes listas para guardar los resultados de la integración
        epp = []
        ecp = []
        etap = []
        app = []
        
        def f(x): # Función a integrar
            # A partir del parámetro x tomamos las diferentes variables del problema
            ep = x[0] # Excentricidad del planeta interior
            ec = x[1] # Excentricidad del planeta exterior
            eta = x[2] # Diferencia entre las longitudes del periihelio de ambos planetas (w2 - w1)
            ap = x[3] # Semieje del planeta interior
            
            np = sqrt(mu/ap**3) # Movimiento medio del planeta interior
            epsilonc = sqrt(1 - ec**2) # Lo definimos para simplificar la escritura de las siguientes ecuaciones
            
            # Constantes que surgen a partir de las aproximaciones hechas 
            Wo = 15/16 * np * (ap/ac)**4 * (mc/ms) * epsilonc**-5
            Wc = 15/16 * nc * (ap/ac)**3 * (mp/ms) * epsilonc**-4
            Wt = 21/2 * np * (kp/Qp) * (ms/mp) * (Rp/ap)**5
            Wq = 3/4 * np * (ap/ac)**3 * mc/ms * epsilonc**-3 * (1 - sqrt(ap/ac) * mp/mc * epsilonc**-1 + gamma * epsilonc**3)
            
            # Las diferentes ecuaciones diferenciales a integrar
            dep = -Wo * ec * sin(eta) - Wt * ep
            dec = Wc * ep * sin(eta)
            deta = Wq - Wo * ec/ep * cos(eta)
            dap = -(2/3) * np * (ap**-4) * ((2 + 46 * ep**2) * s + 7 * ep**2 *p)
            
            return array([dep,dec,deta,dap],float) # Devolvemos los valores de las ecuaciones diferenciales (derivadas)
        
        # Comenzamos a integrar utilizando el método propuesto (rk4)
        for t in tp:
            epp.append(x[0])
            ecp.append(x[1])
            etap.append(x[2])
            app.append(x[3])

            k1 = h * f(x)
            k2 = h * f(x + 0.5 * k1)
            k3 = h * f(x + 0.5 * k2)
            k4 = h * f(x + k3)
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        return array([epp,ecp,app,tp],float) # Devolvemos las listas que nos interesan para el posterior análisis y gráficas
    
    epp, ecp, app, tp = rk4(a,b,N,x) # Llamamos a la función que integra
    
    return epp, ecp, app, tp # Devolvemos las listas que nos interesan

def ev_e(e,tp,titulo,color,e0):
    
    # Función que grafica la evolución de la excentricidad
    
    # e: lista de excentricidades para un sistema dado
    # tp: array con los tiempos correspondientes a la excentricidad
    # titulo: título de la gráfica
    # color: color de la gráfica
    
    plot(tp,e,label='e0 = %.2f'%e0, color = color, linewidth=0.5)
    title(titulo)
    xlabel('Tiempo (años)')
    ylabel('e')
    legend()
    ticklabel_format(useOffset=False)
  
def ev_a(a,tp,titulo,color,e):
        
    # Función que grafica la evolución del semieje mayor
    
    # a: lista de semiejes para un sistema dado
    # tp: array con los tiempos correspondientes a la excentricidad
    # titulo: título de la gráfica
    # color: color de la gráfica
    # e: condición inicial de la excentricidad 
    
    plot(tp,a,label = 'e0 = %.2f'%e, color = color)
    xlabel('Tiempo (años)')
    ylabel('Semieje (UA)')
    title(titulo)
    legend()
    ticklabel_format(useOffset=False)
    
###############################################################################
########################## Definición de constantes ###########################
###############################################################################

# Trabajamos con unidades solares para masa, unidades astronómicas para distancia y años para tiempo

# Masas en masas solares
ms = 1
mj = ms * 9.547919 * 1e-4 # datos de Júpiter
mt = ms * 1/333000 # datos de la Tierra

# Radios en unidades astronómicas
Rs = 0.00465047
Rj = 0.10045 * Rs # datos de Júpiter
Rt = 0.00914927 * Rs # datos de la Tierra

G = 4*pi**2 # Constante de gravitación en ua, años y masas solares

# Constantes referentes a la disipación
kp = 1
kds = 1
kdp = 1

Qs = 1e7 # Constante de disipación para una estrella como el Sol

gamma = 0 # Constante referente a la relatividad, no asumo efectos relativistas 

a = 0 # Tiempo incial de integración, es el mismo para todos los casos

###############################################################################
#################### Definición los diferentes sistemas #######################
###############################################################################

# Vamos a tomar diferentes condiciones iniciales de la excentricidad del planeta interior
# Tomamos 0.2, 0.4 y 0.6
# Recordamos la forma del vector con condiciones iniciales x:
# x = array([ep, ec, eta, ac],float)

# Estas funciones reciben como parámetros el tiempo final y la cantidad de iteraciones deseadas

#------------------------------------------------------------------------------
# Sistema de Super - Tierra y Super - Tierra
#------------------------------------------------------------------------------

def stst(b,N):
    
    ac = 0.046 # Semieje del planeta compañero
    
    Rp = 1.58 * Rt # Definimos el radio del planeta interior
    mp = mt * 8  # Definimos la masa del planeta interior
    mc = mt * 13.6 # Definimos la masa del planeta compañero
    Qp = 1e3 # Para Super Tierras 
      
    x = array([0.2,0.4,0.01,0.017],float) # Condiciones iniciales para el primer caso, ep = 0.2
    
    epp, ecp, app, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # Calculamos toda la evolución
    
    x = array([0.4,0.4,0.01,0.017],float) # Condiciones iniciales para el segundo caso, ep = 0.4
    
    epp2, ecp2, app2, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp)
    
    x = array([0.6,0.4,0.01,0.017],float) # Condiciones iniciales para el tercer caso, ep = 0.5
    
    epp3, ecp3, app3, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp)
    
    figure(dpi=300)
    ev_e(epp, tp, '', 'salmon', epp[0])
    ev_e(epp2, tp, '', 'cornflowerblue', epp2[0])
    ev_e(epp3, tp, 'Evolución de e para el planeta interno,\n para un sistema Super-Tierra (int) y Super-Tierra (ext)', 'darkturquoise', epp3[0])
    show()
    
    figure(dpi=300)
    ev_e(ecp, tp, '', 'salmon', epp[0])
    ev_e(ecp2, tp, '', 'cornflowerblue', epp2[0])
    ev_e(ecp3, tp, 'Evolución de e para el planeta externo,\n para un sistema Super-Tierra (int) y Super-Tierra (ext)', 'darkturquoise', epp3[0])
    show()
    
    figure(dpi=300)
    ev_a(app, tp, '', 'salmon', epp[0])
    ev_a(app2, tp, '','cornflowerblue', epp2[0])
    ev_a(app3, tp, 'Evolución de a para el planeta interno,\n para un sistema Super-Tierra (int) y Super-Tierra (ext)', 'darkturquoise', epp3[0])
    show()
    
    fig, (ax1, ax2, ax3) = subplots(3, 1, dpi=300, sharex=True)
    fig.suptitle('Evolución de e para ambos planetas')
    ax1.plot(tp, epp, color = 'salmon', label='Planeta interno', linewidth=0.5)
    ax1.plot(tp,ecp, color = 'cornflowerblue', label = 'Planeta externo', linewidth=0.5)
    ax1.set_title(r'$e_{pi}$ = 0.2 $e_{ci} = 0.4$')
    ax2.plot(tp, epp2, color = 'salmon', label='Planeta interno', linewidth=0.5)
    ax2.plot(tp,ecp2, color = 'cornflowerblue', label = 'Planeta externo', linewidth=0.5)
    ax2.set_title(r'$e_{pi}$ = 0.4 $e_{ci} = 0.4$')
    ax3.plot(tp, epp3, color = 'salmon', label='Planeta interno', linewidth=0.5)
    ax3.plot(tp,ecp3, color = 'cornflowerblue', label = 'Planeta externo', linewidth=0.5)
    ax3.set_title(r'$e_{pi}$ = 0.6 $e_{ci} = 0.4$')
    legend()
    fig.tight_layout()

#------------------------------------------------------------------------------
# Sistema de Super - Tierra y Júpiter
#------------------------------------------------------------------------------

def stj(b,N):

    ac = 1 # Semieje del planeta compañero
    
    Rp = 5**(1/3) * Rt # Definimos el radio del planeta interior
    mp = mt * 5 # Definimos la masa del planeta interior
    mc = mj # Definimos la masa del planeta compañero
    Qp = 1e3 # Para Super Tierras 
    
    x = array([0.2,0.4,0.01,0.04],float) # Condiciones iniciales para el primer caso, ep = 0.2
    
    epp, ecp, app, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # Calculamos toda la evolución
    
    x = array([0.4,0.4,0.01,0.04],float) # Condiciones iniciales para el segundo caso, ep = 0.4
    
    epp2, ecp2, app2, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp)
    
    x = array([0.6,0.4,0.01,0.04],float) # Condiciones iniciales para el tercer caso, ep = 0.5
    
    epp3, ecp3, app3, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp)
    
    figure(dpi=300)
    ev_e(epp, tp, '', 'salmon', epp[0])
    ev_e(epp2, tp, '', 'cornflowerblue', epp2[0])
    ev_e(epp3, tp, 'Evolución de e para el planeta interno,\n para un sistema Super-Tierra (int) y Júpiter (ext)', 'darkturquoise', epp3[0])
    show()
    
    figure(dpi=300)
    ev_e(ecp, tp, '', 'salmon', epp[0])
    ev_e(ecp2, tp, '', 'cornflowerblue', epp2[0])
    ev_e(ecp3, tp, 'Evolución de e para el planeta externo,\n para un sistema Super-Tierra (int) y Júpiter (ext)', 'darkturquoise', epp3[0])
    show()
    
    figure(dpi=300)
    ev_a(app, tp, '', 'salmon', epp[0])
    ev_a(app2, tp, '', 'cornflowerblue', epp2[0])
    ev_a(app3, tp, 'Evolución de a para el planeta interno,\n para un sistema Super-Tierra (int) y Júpiter (ext)', 'darkturquoise', epp3[0])
    show()
    
    fig, (ax1, ax2, ax3) = subplots(3, 1, dpi=300, sharex=True)
    fig.suptitle('Evolución de e para ambos planetas')
    ax1.plot(tp, epp, color = 'salmon', label='Planeta interno', linewidth=0.5)
    ax1.plot(tp,ecp, color = 'cornflowerblue', label = 'Planeta externo', linewidth=1)
    ax1.set_title(r'$e_{pi}$ = 0.2 $e_{ci} = 0.4$')
    ax2.plot(tp, epp2, color = 'salmon', label='Planeta interno', linewidth=0.5)
    ax2.plot(tp,ecp2, color = 'cornflowerblue', label = 'Planeta externo', linewidth=1)
    ax2.set_title(r'$e_{pi}$ = 0.4 $e_{ci} = 0.4$')
    ax3.plot(tp, epp3, color = 'salmon', label='Planeta interno', linewidth=0.5)
    ax3.plot(tp,ecp3, color = 'cornflowerblue', label = 'Planeta externo', linewidth=1)
    ax3.set_title(r'$e_{pi}$ = 0.6 $e_{ci} = 0.4$')
    legend()
    fig.tight_layout()
    
#------------------------------------------------------------------------------
# Sistema de Júpiter y Neptuno
#------------------------------------------------------------------------------

def jn(N): # Dentro de la función tratamos con diferentes límites temporales, entonces solo recibe como parámetro la cantidad de iteraciones N
    
    ac = 0.5 # Semieje del planeta compañero
    
    Rp = Rj * 2
    mc = 0.0539531012 * mj # Masa de Neptuno (compañero)
    mp = mj * 0.46 # Masa de 51 Pegasi b (planeta interior)
    Qp = 1e5 # Para júpiteres 
    
    b = 1e8 # 100 millones de años
    
    x = array([0.2,0.4,0.01,0.05],float) # Condiciones iniciales para el primer caso, ep = 0.2
    
    epp, ecp, app, tp1 = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp)
    
    x = array([0.4,0.4,0.01,0.05],float) # Condiciones iniciales para el segundo caso, ep = 0.4
    
    epp2, ecp2, app2, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp)
    
    x = array([0.6,0.4,0.01,0.05],float) # Condiciones iniciales para el tercer caso, ep = 0.6
    
    epp3, ecp3, app3, tp1 = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp)
    
    b=1e7 # 10 millones de años
    
    x = array([0.2,0.4,0.01,0.05],float)
    
    epp4, ecp4, app4, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # ep = 0.2
    
    x = array([0.4,0.4,0.01,0.05],float)
    
    epp5, ecp5, app5, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # ep = 0.4
    
    x = array([0.6,0.4,0.01,0.05],float)
    
    epp6, ecp6, app6, tp2 = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # ep = 0.6
    
    b=1e6 # 1 millón de años
    
    x = array([0.2,0.4,0.01,0.05],float)
    
    epp7, ecp7, app7, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # ep = 0.2
    
    x = array([0.4,0.4,0.01,0.05],float)
    
    epp8, ecp8, app8, tp = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # ep = 0.4
    
    x = array([0.6,0.4,0.01,0.05],float)
    
    epp9, ecp9, app9, tp3 = evolucion(a,b,N,x,mp,mc,Rp,ac,Qp) # ep = 0.6
    
    fig, (ax1, ax2, ax3) = subplots(3, 1, dpi=300)
    fig.suptitle('Evolución de e para el planeta interno')
    ax1.plot(tp1, epp, color = 'salmon',  linewidth=0.5)
    ax1.plot(tp1,epp2, color = 'cornflowerblue',  linewidth=0.5)
    ax1.plot(tp1,epp3, color = 'darkturquoise',  linewidth=0.5)
    ax2.plot(tp2, epp4, color = 'salmon', linewidth=0.5)
    ax2.plot(tp2,epp5, color = 'cornflowerblue',  linewidth=0.5)
    ax2.plot(tp2,epp6, color = 'darkturquoise',  linewidth=0.5)
    ax3.plot(tp3, epp7, color = 'salmon', label='e0 = 0.2', linewidth=0.5)
    ax3.plot(tp3,epp8, color = 'cornflowerblue', label = 'e0 = 0.4', linewidth=0.5)
    ax3.plot(tp3,epp9, color = 'darkturquoise', label = 'e0 = 0.6', linewidth=0.5)
    ylabel('e')
    xlabel('Tiempo (años)')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  
    fig.legend( loc="lower center", ncol=3)   
        
    fig, (ax1, ax2, ax3) = subplots(3, 1, dpi=300)
    fig.suptitle('Evolución de e para el planeta externo')
    ax1.plot(tp1, ecp, color = 'salmon', linewidth=0.5)
    ax1.plot(tp1,ecp2, color = 'cornflowerblue',  linewidth=0.5)
    ax1.plot(tp1,ecp3, color = 'darkturquoise',  linewidth=0.5)
    ax2.plot(tp2, ecp4, color = 'salmon',  linewidth=0.5)
    ax2.plot(tp2,ecp5, color = 'cornflowerblue',  linewidth=0.5)
    ax2.plot(tp2,ecp6, color = 'darkturquoise',  linewidth=0.5)
    ax3.plot(tp3, ecp7, color = 'salmon', label='e0 = 0.2', linewidth=0.5)
    ax3.plot(tp3,ecp8, color = 'cornflowerblue', label = 'e0 = 0.4', linewidth=0.5)
    ax3.plot(tp3,ecp9, color = 'darkturquoise', label = 'e0 = 0.6', linewidth=0.5)
    ylabel('e')
    xlabel('Tiempo (años)')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  
    fig.legend( loc="lower center", ncol=3)
    
    fig, (ax1, ax2, ax3) = subplots(3, 1, dpi=300)
    fig.suptitle('Evolución de a para el planeta interno')
    ax1.plot(tp1, app, color = 'salmon', linewidth=0.5)
    ax1.plot(tp1,app2, color = 'cornflowerblue',  linewidth=0.5)
    ax1.plot(tp1,app3, color = 'darkturquoise', linewidth=0.5)
    ax2.plot(tp2, app4, color = 'salmon',  linewidth=0.5)
    ax2.plot(tp2,app5, color = 'cornflowerblue',  linewidth=0.5)
    ax2.plot(tp2,app6, color = 'darkturquoise',  linewidth=0.5)
    ax3.plot(tp3, app7, color = 'salmon', label='e0 = 0.2', linewidth=0.5)
    ax3.plot(tp3,app8, color = 'cornflowerblue', label = 'e0 = 0.4', linewidth=0.5)
    ax3.plot(tp3,app9, color = 'darkturquoise', label = 'e0 = 0.6', linewidth=0.5)
    ylabel('Semieje (UA)')
    xlabel('Tiempo (años)')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  
    fig.legend( loc="lower center", ncol=3) 
    
    fig, (ax1, ax2, ax3) = subplots(3, 1, dpi=300, sharex = True)
    fig.suptitle('Evolución de e para ambos planetas')
    ax1.plot(tp2, epp4, color = 'salmon', linewidth=0.5)
    ax1.plot(tp2,ecp4, color = 'cornflowerblue',  linewidth=0.5)
    ax1.set_title(r'$e_{pi}$ = 0.2 $e_{ci} = 0.4$')
    ax2.plot(tp2, epp5, color = 'salmon',  linewidth=0.5)
    ax2.plot(tp2,ecp5, color = 'cornflowerblue',  linewidth=0.5)
    ax2.set_title(r'$e_{pi}$ = 0.4 $e_{ci} = 0.4$')
    ax3.plot(tp2, epp6, color = 'salmon', label='Planeta interno', linewidth=0.5)
    ax3.plot(tp2,ecp6, color = 'cornflowerblue', label = 'Planeta externo', linewidth=0.5)
    ax3.set_title(r'$e_{pi}$ = 0.6 $e_{ci} = 0.4$')
    ylabel('e')
    xlabel('Tiempo (años)')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)  
    fig.legend( loc="lower center", ncol=3)

###############################################################################
##################### Graficamos los distintos sistemas #######################
###############################################################################

stst(2e6, 1e5)
stj(1.2e8, 1e5)
jn(1e5)