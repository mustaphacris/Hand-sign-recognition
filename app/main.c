// Inclusion des bibliothèques système nécessaires pour les périphériques et l'I2C.
#include "system.h"
#include "sys/alt_stdio.h"
#include "alt_types.h"
#include "io.h"
#include "unistd.h"
#include "sys/alt_sys_init.h"
#include "opencores_i2c.h"
#include "opencores_i2c_regs.h"
#include "altera_avalon_pio_regs.h"
#include "sys/alt_irq.h"
#include "altera_avalon_timer_regs.h"
#include "altera_avalon_timer.h"


// Déclaration des constantes liées à l'ADXL345 et aux paramètres matériels.
#define ALT_ADDR 0x1d  // Adresse de l'ADXL345
#define SPEED 100000   // Vitesse de communication I2C
#define ACT_INACT_CTL 0x27
#define POWER_CTL 0x2D
#define DATA_FORMAT 0x31


#define X0 0x32  // Adresse du registre X-axis LSB
#define X1 0x33  // Adresse du registre X-axis MSB
#define Y0 0x34  // Adresse du registre Y-axis LSB
#define Y1 0x35  // Adresse du registre Y-axis MSB
#define Z0 0x36  // Adresse du registre Z-axis LSB
#define Z1 0x37  // Adresse du registre Z-axis MSB


#define OffsetX 0x1E  // Offset X
#define OffsetY 0x1F  // Offset Y
#define OffsetZ 0x20  // Offset Z


#define CALIBX 0   // Calibration offset X
#define CALIBY 2   // Calibration offset Y
#define CALIBZ 18  // Calibration offset Z


// Variables globales pour l'axe sélectionné.
int A0, A1;


// Fonction de lecture depuis un périphérique I2C.
int I2C_READ(int base, int addr) {

    int data = 0;
    I2C_start(base, ALT_ADDR, 0);  // Initialisation d'une écriture
    I2C_write(base, addr, 0);     // Écriture de l'adresse
    I2C_start(base, ALT_ADDR, 1); // Initialisation d'une lecture
    data = I2C_read(base, 1);     // Lecture du registre

    return data;  // Renvoie la donnée lue
}



// Fonction d'écriture dans un périphérique I2C.
void I2C_Write(int base, int addr, int value) {
    I2C_start(base, ALT_ADDR, 0);  // Démarrage de la communication en écriture
    I2C_write(base, addr, 0);      // Envoie de l'adresse
    I2C_write(base, value, 1);     // Envoie de la valeur
}



// Calibration de l'ADXL345 en réglant les offsets des axes.
void calibration_ADXL345(){
    I2C_Write(OPENCORES_I2C_0_BASE, OffsetZ, CALIBZ);
    usleep(100000);  // Temporisation pour stabiliser l'I2C

    I2C_Write(OPENCORES_I2C_0_BASE, OffsetX, CALIBX);
    usleep(100000);

    I2C_Write(OPENCORES_I2C_0_BASE, OffsetY, CALIBY);
    usleep(100000);
}



// Convertit un entier en une séquence pour l'affichage sur les 7 segments.
void int_to_seg(int nbr) {
    int i = 0;
    int tab[5] = {0, 0, 0, 0, 0};  // Tableau pour stocker chaque chiffre
    
    // Affiche un signe si nécessaire
    if (nbr < 0) {
        IOWR_ALTERA_AVALON_PIO_DATA(SEG5_BASE, 0b0111111);  // Signe négatif
        nbr = nbr * -1;  // Convertit en valeur absolue
    } else {
        IOWR_ALTERA_AVALON_PIO_DATA(SEG5_BASE, 0b1000000);  // Aucun signe
    }
    
    // Décompose l'entier en chiffres
    while (nbr >= 10) {
        tab[i++] = nbr % 10;
        nbr /= 10;
    }
    tab[i] = nbr;

    // Écriture dans les registres des 7 segments
    IOWR_ALTERA_AVALON_PIO_DATA(SEG0_BASE, tab[0]);
    IOWR_ALTERA_AVALON_PIO_DATA(SEG1_BASE, tab[1]);
    IOWR_ALTERA_AVALON_PIO_DATA(SEG2_BASE, tab[2]);
    IOWR_ALTERA_AVALON_PIO_DATA(SEG3_BASE, tab[3]);
    IOWR_ALTERA_AVALON_PIO_DATA(SEG4_BASE, tab[4]);
}



// Affiche les données des axes sur les 7 segments
void affichage_XYZ(int A0, int A1) {
    int a0 = I2C_READ(OPENCORES_I2C_0_BASE, A0);
    int a1 = I2C_READ(OPENCORES_I2C_0_BASE, A1);
    int a = (a1 << 8) | a0;  // Combine MSB et LSB
    a = (short)a;  // Conversion en entier signé
    a = a * 3.9;   // Mise à l'échelle
    int_to_seg(a);  // Affichage
}



// Affiche les données des axes sur la console UART.
void UART_affichage() {

    int x0 = I2C_READ(OPENCORES_I2C_0_BASE, X0);
    int x1 = I2C_READ(OPENCORES_I2C_0_BASE, X1);

    int y0 = I2C_READ(OPENCORES_I2C_0_BASE, Y0);
    int y1 = I2C_READ(OPENCORES_I2C_0_BASE, Y1);

    int z0 = I2C_READ(OPENCORES_I2C_0_BASE, Z0);
    int z1 = I2C_READ(OPENCORES_I2C_0_BASE, Z1);
    

    alt_printf("X= %x, Y= %x, Z= %x\n", (x1 << 8) | x0, (y1 << 8) | y0, (z1 << 8) | z0);
    alt_printf("--------------------\n");
}



// Initialisation de l'ADXL345.
void init_ADXL345() {

    
    I2C_init(OPENCORES_I2C_0_BASE, ALT_CPU_FREQ, SPEED);
    if (I2C_start(OPENCORES_I2C_0_BASE, ALT_ADDR, 0) == 0) {
        alt_printf("Initialisation done\n");
    }

    
    I2C_Write(OPENCORES_I2C_0_BASE, POWER_CTL, 0x08);  // Mise en marche
    usleep(100000);

    
    alt_printf("POWER_CTL = %x\n",I2C_READ(OPENCORES_I2C_0_BASE,POWER_CTL)); //Lecture POWER_CTL

    
    I2C_Write(OPENCORES_I2C_0_BASE, DATA_FORMAT, 0x07);  // Plage de ±16g
    usleep(100000);

    
    alt_printf("DATA_FORMAT = %x\n\n",I2C_READ(OPENCORES_I2C_0_BASE,DATA_FORMAT)); //Lecture DATA_FORMAT

    A0 = X0;  // Configuration initiale pour X
    A1 = X1;
}



// Gère les interruptions déclenchées par les boutons (KEY).
static void key_interrupt(void *Context, alt_u32 id) {

    alt_printf("interruption par le bouton\n\n");

    switch (A0) {
        case X0:
	    alt_printf("Basculer vers Y\n\n");
            A0 = Y0;
            A1 = Y1;
            break;
        case Y0:
            alt_printf("Basculer vers Z\n\n");
            A0 = Z0;
            A1 = Z1;
            break;
        case Z0:
	    alt_printf("Basculer vers X\n\n");
	    A0 = X0;
	    A1 = X1;
	    break;
        default:
            alt_printf("Basculer vers Z\n\n");
            A0 = X0;
            A1 = X1;
            break;
    }

    IOWR_ALTERA_AVALON_PIO_EDGE_CAP(KEY_BASE, 0b1);  // Reset du signal d'interruption
}


// Initialise l'interruption des boutons.
void init_key_interrupt() {

    IOWR_ALTERA_AVALON_PIO_IRQ_MASK(KEY_BASE, 0b1);  // Active l'interruption, applique un mask 0b1 afin d'activer les boutons
    IOWR_ALTERA_AVALON_PIO_EDGE_CAP(KEY_BASE,0b1);   // Active la detection des boutons
    if (alt_irq_register(KEY_IRQ, NULL, key_interrupt) != 0) {
        
    }
}



// Gère les interruptions liées au timer.
static void timer_interrupt(void *Context, alt_u32 id) {

    alt_printf("interruption liee au timer\n");
    UART_affichage();    // Affichage des données UART
    affichage_XYZ(A0, A1);  // Mise à jour de l'affichage 7-segment
    IOWR_ALTERA_AVALON_TIMER_STATUS(TIMER_0_BASE, 0b01);  // Reset du timer
}



// Initialise le timer pour l'interruption.
void init_timer_interrupt() {
    if (alt_irq_register(TIMER_0_IRQ, NULL, timer_interrupt) != 0) {
        
    }
}



// Fonction principale.
int main() {

    init_ADXL345();
    calibration_ADXL345();
    init_key_interrupt();
    init_timer_interrupt();

    while (1) {
        // Boucle infinie, les interruptions prennent le relais.
    }
}
