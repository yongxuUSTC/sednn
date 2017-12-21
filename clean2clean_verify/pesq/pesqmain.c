/*****************************************************************************

Perceptual Evaluation of Speech Quality (PESQ)
ITU-T Recommendation P.862.
Version 1.2 - 2 August 2002.

              ****************************************
              PESQ Intellectual Property Rights Notice
              ****************************************

DEFINITIONS:
------------
For the purposes of this Intellectual Property Rights Notice
the terms 'Perceptual Evaluation of Speech Quality Algorithm'
and 'PESQ Algorithm' refer to the objective speech quality
measurement algorithm defined in ITU-T Recommendation P.862;
the term 'PESQ Software' refers to the C-code component of P.862. 

NOTICE:
-------
All copyright, trade marks, trade names, patents, know-how and
all or any other intellectual rights subsisting in or used in
connection with including all algorithms, documents and manuals
relating to the PESQ Algorithm and or PESQ Software are and remain
the sole property in law, ownership, regulations, treaties and
patent rights of the Owners identified below. The user may not
dispute or question the ownership of the PESQ Algorithm and
or PESQ Software.

OWNERS ARE:
-----------

1.	British Telecommunications plc (BT), all rights assigned
      to Psytechnics Limited
2.	Royal KPN NV, all rights assigned to OPTICOM GmbH

RESTRICTIONS:
-------------

The user cannot:

1.	alter, duplicate, modify, adapt, or translate in whole or in
      part any aspect of the PESQ Algorithm and or PESQ Software
2.	sell, hire, loan, distribute, dispose or put to any commercial
      use other than those permitted below in whole or in part any
      aspect of the PESQ Algorithm and or PESQ Software

PERMITTED USE:
--------------

The user may:

1.	Use the PESQ Software to:
      i)   understand the PESQ Algorithm; or
      ii)  evaluate the ability of the PESQ Algorithm to perform
           its intended function of predicting the speech quality
           of a system; or
      iii) evaluate the computational complexity of the PESQ Algorithm,
           with the limitation that none of said evaluations or its
           results shall be used for external commercial use.

2.	Use the PESQ Software to test if an implementation of the PESQ
      Algorithm conforms to ITU-T Recommendation P.862.

3.	With the prior written permission of both Psytechnics Limited
      and OPTICOM GmbH, use the PESQ Software in accordance with the
      above Restrictions to perform work that meets all of the following
      criteria:
      i)    the work must contribute directly to the maintenance of an
            existing ITU recommendation or the development of a new ITU
            recommendation under an approved ITU Study Item; and
      ii)   the work and its results must be fully described in a
            written contribution to the ITU that is presented at a formal
            ITU meeting within one year of the start of the work; and
      iii)  neither the work nor its results shall be put to any
            commercial use other than making said contribution to the ITU.
            Said permission will be provided on a case-by-case basis.


ANY OTHER USE OR APPLICATION OF THE PESQ SOFTWARE AND/OR THE PESQ
ALGORITHM WILL REQUIRE A PESQ LICENCE AGREEMENT, WHICH MAY BE OBTAINED
FROM EITHER OPTICOM GMBH OR PSYTECHNICS LIMITED. 

EACH COMPANY OFFERS OEM LICENSE AGREEMENTS, WHICH COMBINE OEM
IMPLEMENTATIONS OF THE PESQ ALGORITHM TOGETHER WITH A PESQ PATENT LICENSE
AGREEMENT. PESQ PATENT-ONLY LICENSE AGREEMENTS MAY BE OBTAINED FROM OPTICOM.


***********************************************************************
*  OPTICOM GmbH                    *  Psytechnics Limited             *
*  Am Weichselgarten 7,            *  Fraser House, 23 Museum Street, *
*  D- 91058 Erlangen, Germany      *  Ipswich IP1 1HN, England        *
*  Phone: +49 (0) 9131 691 160     *  Phone: +44 (0) 1473 261 800     *
*  Fax:   +49 (0) 9131 691 325     *  Fax:   +44 (0) 1473 261 880     *
*  E-mail: info@opticom.de,        *  E-mail: info@psytechnics.com,   *
*  www.opticom.de                  *  www.psytechnics.com             *
***********************************************************************

Further information is also available from www.pesq.org

*****************************************************************************/

#include <stdio.h>
#include <math.h>
#include "pesq.h"
#include "dsp.h"

#define ITU_RESULTS_FILE          "_pesq_itu_results.txt"
#define SIMPLE_RESULTS_FILE       "_pesq_results.txt"


int main (int argc, const char *argv []);
void usage (void);
void pesq_measure (SIGNAL_INFO * ref_info, SIGNAL_INFO * deg_info,
    ERROR_INFO * err_info, long * Error_Flag, char ** Error_Type);

void usage (void) {
    printf ("Usage:\n");
    printf (" PESQ HELP               Displays this text\n");
    printf (" PESQ [options] ref deg [smos] [cond]\n");
    printf (" Run model on reference ref and degraded deg\n");
    printf ("\n");
    printf ("Options: +8000 +16000 +swap\n");
    printf (" Sample rate - No default. Must select either +8000 or +16000.\n");
    printf (" Swap byte order - machine native format by default. Select +swap for byteswap.\n");
    printf ("\n");
    printf (" [smos] is an optional number copied to %s\n", ITU_RESULTS_FILE);
    printf (" [cond] is an optional condition number copied to %s\n", ITU_RESULTS_FILE);
    printf (" smos must always precede cond. However, both may be omitted.");
    printf ("\n");
    printf ("File names, smos, cond may not begin with a + character.\n");
    printf ("\n");
    printf ("Files with names ending .wav or .WAV are assumed to have a 44-byte header, which");
    printf (" is automatically skipped.  All other file types are assumed to have no header.\n");
}

int main (int argc, const char *argv []) {
    int  arg;
    int  names = 0;
    long sample_rate = -1;
    
    SIGNAL_INFO ref_info;
    SIGNAL_INFO deg_info;
    ERROR_INFO err_info;

    long Error_Flag = 0;
    char * Error_Type = "Unknown error type.";

    if (Error_Flag == 0) {
        printf("Perceptual Evaluation of Speech Quality (PESQ) - ITU-T Recommendation P.862.\n");
        printf("Version 1.2 - 2 August 2002.\n");
        printf("\n");
        printf("PESQ Intellectual Property Rights Notice.\n");
        printf("\n");
        printf("DEFINITIONS:\n");
        printf("For the purposes of this Intellectual Property Rights Notice the terms\n");
        printf("'Perceptual Evaluation of Speech Quality Algorithm' and 'PESQ Algorithm'\n");
        printf("refer to the objective speech quality measurement algorithm defined in ITU-T\n");
        printf("Recommendation P.862; the term 'PESQ Software' refers to the C-code component\n");
        printf("of P.862.\n");
        printf("\n");
        printf("NOTICE:\n");
        printf("All copyright, trade marks, trade names, patents, know-how and all or any other\n");
        printf("intellectual rights subsisting in or used in connection with including all\n");
        printf("algorithms, documents and manuals relating to the PESQ Algorithm and or PESQ\n");
        printf("Software are and remain the sole property in law, ownership, regulations,\n");
        printf("treaties and patent rights of the Owners identified below. The user may not\n");
        printf("dispute or question the ownership of the PESQ Algorithm and or PESQ Software.\n");
        printf("\n");
        printf("OWNERS ARE:\n");
        printf("1.	British Telecommunications plc (BT), all rights assigned\n");
        printf("      to Psytechnics Limited\n");
        printf("2.	Royal KPN NV, all rights assigned to OPTICOM GmbH\n");
        printf("\n");
        printf("RESTRICTIONS:\n");
        printf("The user cannot:\n");
        printf("1.	alter, duplicate, modify, adapt, or translate in whole or in\n");
        printf("      part any aspect of the PESQ Algorithm and or PESQ Software\n");
        printf("2.	sell, hire, loan, distribute, dispose or put to any commercial\n");
        printf("      use other than those permitted below in whole or in part any\n");
        printf("      aspect of the PESQ Algorithm and or PESQ Software\n");
        printf("\n");
        printf("PERMITTED USE:\n");
        printf("The user may:\n");
        printf("1.	Use the PESQ Software to:\n");
        printf("      i)   understand the PESQ Algorithm; or\n");
        printf("      ii)  evaluate the ability of the PESQ Algorithm to perform its intended\n");
        printf("           function of predicting the speech quality of a system; or\n");
        printf("      iii) evaluate the computational complexity of the PESQ Algorithm,\n");
        printf("           with the limitation that none of said evaluations or its\n");
        printf("           results shall be used for external commercial use.\n");
        printf("2.	Use the PESQ Software to test if an implementation of the PESQ\n");
        printf("      Algorithm conforms to ITU-T Recommendation P.862.\n");
        printf("3.	With the prior written permission of both Psytechnics Limited and\n");
        printf("      OPTICOM GmbH, use the PESQ Software in accordance with the above\n");
        printf("      Restrictions to perform work that meets all of the following criteria:\n");
        printf("      i)    the work must contribute directly to the maintenance of an\n");
        printf("            existing ITU recommendation or the development of a new ITU\n");
        printf("            recommendation under an approved ITU Study Item; and\n");
        printf("      ii)   the work and its results must be fully described in a\n");
        printf("            written contribution to the ITU that is presented at a formal\n");
        printf("            ITU meeting within one year of the start of the work; and\n");
        printf("      iii)  neither the work nor its results shall be put to any\n");
        printf("            commercial use other than making said contribution to the ITU.\n");
        printf("            Said permission will be provided on a case-by-case basis.\n");
        printf("\n");
        printf("ANY OTHER USE OR APPLICATION OF THE PESQ SOFTWARE AND/OR THE PESQ ALGORITHM\n");
        printf("WILL REQUIRE A PESQ LICENCE AGREEMENT, WHICH MAY BE OBTAINED FROM EITHER\n");
        printf("OPTICOM GMBH OR PSYTECHNICS LIMITED. \n");
        printf("\n");
        printf("EACH COMPANY OFFERS OEM LICENSE AGREEMENTS, WHICH COMBINE OEM\n");
        printf("IMPLEMENTATIONS OF THE PESQ ALGORITHM TOGETHER WITH A PESQ PATENT LICENSE\n");
        printf("AGREEMENT. PESQ PATENT-ONLY LICENSE AGREEMENTS MAY BE OBTAINED FROM OPTICOM.\n");
        printf("\n");
        printf("***********************************************************************\n");
        printf("*  OPTICOM GmbH                    *  Psytechnics Limited             *\n");
        printf("*  Am Weichselgarten 7,            *  Fraser House, 23 Museum Street, *\n");
        printf("*  D- 91058 Erlangen, Germany      *  Ipswich IP1 1HN, England        *\n");
        printf("*  Phone: +49 (0) 9131 691 160     *  Phone: +44 (0) 1473 261 800     *\n");
        printf("*  Fax:   +49 (0) 9131 691 325     *  Fax:   +44 (0) 1473 261 880     *\n");
        printf("*  E-mail: info@opticom.de,        *  E-mail: info@psytechnics.com,   *\n");
        printf("*  www.opticom.de                  *  www.psytechnics.com             *\n");
        printf("***********************************************************************\n");
        printf("\n");

        if (argc < 3){
            usage ();
            return 0;                                                                  
        } else {

            strcpy (ref_info.path_name, "");
            ref_info.apply_swap = 0;
            strcpy (deg_info.path_name, "");
            deg_info.apply_swap = 0;
            err_info. subj_mos = 0;
            err_info. cond_nr = 0;

            for (arg = 1; arg < argc; arg++) {
                if (argv [arg] [0] == '+') {
                    if (strcmp (argv [arg], "+swap") == 0) {
                        ref_info.apply_swap = 1;
                        deg_info.apply_swap = 1;
                    } else {
                        if (strcmp (argv [arg], "+16000") == 0) {
                            sample_rate = 16000L;
                        } else {
                            if (strcmp (argv [arg], "+8000") == 0) {
                                sample_rate = 8000L;
                            } else {
                                usage ();
                                fprintf (stderr, "Invalid parameter '%s'.\n", argv [arg]);
                                return 1;
                            }
                        }
                    }
                } else {
                    switch (names) {
                        case 0: 
                            strcpy (ref_info.path_name, argv [arg]); 
                            break;
                        case 1: 
                            strcpy (deg_info.path_name, argv [arg]); 
                            break;
                        case 2: 
                            sscanf (argv [arg], "%f", &(err_info. subj_mos)); 
                            break;
                        case 3: 
                            sscanf (argv [arg], "%d", &(err_info. cond_nr)); 
                            break;
                        default:
                            usage ();
                            fprintf (stderr, "Invalid parameter '%s'.\n", argv [arg]);
                            return 1;
                    }
                    names++;
                }
            }

            if (sample_rate == -1) {
                printf ("PESQ Error. Must specify either +8000 or +16000 sample frequency option!\n");
                exit (1);
            }
            
            strcpy (ref_info. file_name, ref_info. path_name);
            if (strrchr (ref_info. file_name, '\\') != NULL) {
                strcpy (ref_info. file_name, 1 + strrchr (ref_info. file_name, '\\'));
            }
            if (strrchr (ref_info. file_name, '/') != NULL) {
                strcpy (ref_info. file_name, 1 + strrchr (ref_info. file_name, '/'));
            }                

            strcpy (deg_info. file_name, deg_info. path_name);
            if (strrchr (deg_info. file_name, '\\') != NULL) {
                strcpy (deg_info. file_name, 1 + strrchr (deg_info. file_name, '\\'));
            }
            if (strrchr (deg_info. file_name, '/') != NULL) {
                strcpy (deg_info. file_name, 1 + strrchr (deg_info. file_name, '/'));
            }                

            select_rate (sample_rate, &Error_Flag, &Error_Type);
            pesq_measure (&ref_info, &deg_info, &err_info, &Error_Flag, &Error_Type);
        }
    }

    if (Error_Flag == 0) {
        printf ("\nPrediction : PESQ_MOS = %.3f\n", (double) err_info.pesq_mos);
        return 0;
    } else {
        printf ("An error of type %d ", Error_Flag);
        if (Error_Type != NULL) {
            printf (" (%s) occurred during processing.\n", Error_Type);
        } else {
            printf ("occurred during processing.\n");
        }

        return 0;
    }
}

double align_filter_dB [26] [2] = {{0.,-500},
                                 {50., -500},
                                 {100., -500},
                                 {125., -500},
                                 {160., -500},
                                 {200., -500},
                                 {250., -500},
                                 {300., -500},
                                 {350.,  0},
                                 {400.,  0},
                                 {500.,  0},
                                 {600.,  0},
                                 {630.,  0},
                                 {800.,  0},
                                 {1000., 0},
                                 {1250., 0},
                                 {1600., 0},
                                 {2000., 0},
                                 {2500., 0},
                                 {3000., 0},
                                 {3250., 0},
                                 {3500., -500},
                                 {4000., -500},
                                 {5000., -500},
                                 {6300., -500},
                                 {8000., -500}}; 


double standard_IRS_filter_dB [26] [2] = {{  0., -200},
                                         { 50., -40}, 
                                         {100., -20},
                                         {125., -12},
                                         {160.,  -6},
                                         {200.,   0},
                                         {250.,   4},
                                         {300.,   6},
                                         {350.,   8},
                                         {400.,  10},
                                         {500.,  11},
                                         {600.,  12},
                                         {700.,  12},
                                         {800.,  12},
                                         {1000., 12},
                                         {1300., 12},
                                         {1600., 12},
                                         {2000., 12},
                                         {2500., 12},
                                         {3000., 12},
                                         {3250., 12},
                                         {3500., 4},
                                         {4000., -200},
                                         {5000., -200},
                                         {6300., -200},
                                         {8000., -200}}; 


#define TARGET_AVG_POWER    1E7

void fix_power_level (SIGNAL_INFO *info, char *name, long maxNsamples) 
{
    long   n = info-> Nsamples;
    long   i;
    float *align_filtered = (float *) safe_malloc ((n + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof (float));    
    float  global_scale;
    float  power_above_300Hz;

    for (i = 0; i < n + DATAPADDING_MSECS  * (Fs / 1000); i++) {
        align_filtered [i] = info-> data [i];
    }
    apply_filter (align_filtered, info-> Nsamples, 26, align_filter_dB);

    power_above_300Hz = (float) pow_of (align_filtered, 
                                        SEARCHBUFFER * Downsample, 
                                        n - SEARCHBUFFER * Downsample + DATAPADDING_MSECS  * (Fs / 1000),
                                        maxNsamples - 2 * SEARCHBUFFER * Downsample + DATAPADDING_MSECS  * (Fs / 1000));

    global_scale = (float) sqrt (TARGET_AVG_POWER / power_above_300Hz); 

    for (i = 0; i < n; i++) {
        info-> data [i] *= global_scale;    
    }

    safe_free (align_filtered);
}

       
void pesq_measure (SIGNAL_INFO * ref_info, SIGNAL_INFO * deg_info,
    ERROR_INFO * err_info, long * Error_Flag, char ** Error_Type)
{
    float * ftmp = NULL;

    ref_info-> data = NULL;
    ref_info-> VAD = NULL;
    ref_info-> logVAD = NULL;
    
    deg_info-> data = NULL;
    deg_info-> VAD = NULL;
    deg_info-> logVAD = NULL;
        
    if ((*Error_Flag) == 0)
    {
        printf ("Reading reference file %s...", ref_info-> path_name);

       load_src (Error_Flag, Error_Type, ref_info);
       if ((*Error_Flag) == 0)
           printf ("done.\n");
    }
    if ((*Error_Flag) == 0)
    {
        printf ("Reading degraded file %s...", deg_info-> path_name);

       load_src (Error_Flag, Error_Type, deg_info);
       if ((*Error_Flag) == 0)
           printf ("done.\n");
    }

    if (((ref_info-> Nsamples - 2 * SEARCHBUFFER * Downsample < Fs / 4) ||
         (deg_info-> Nsamples - 2 * SEARCHBUFFER * Downsample < Fs / 4)) &&
        ((*Error_Flag) == 0))
    {
        (*Error_Flag) = 2;
        (*Error_Type) = "Reference or Degraded below 1/4 second - processing stopped ";
    }

    if ((*Error_Flag) == 0)
    {
        alloc_other (ref_info, deg_info, Error_Flag, Error_Type, &ftmp);
    }

    if ((*Error_Flag) == 0)
    {   
        int     maxNsamples = max (ref_info-> Nsamples, deg_info-> Nsamples);
        float * model_ref; 
        float * model_deg; 
        long    i;
        FILE *resultsFile;

        printf (" Level normalization...\n");            
        fix_power_level (ref_info, "reference", maxNsamples);
        fix_power_level (deg_info, "degraded", maxNsamples);

        printf (" IRS filtering...\n"); 
        apply_filter (ref_info-> data, ref_info-> Nsamples, 26, standard_IRS_filter_dB);
        apply_filter (deg_info-> data, deg_info-> Nsamples, 26, standard_IRS_filter_dB);

        model_ref = (float *) safe_malloc ((ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof (float));
        model_deg = (float *) safe_malloc ((deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof (float));

        for (i = 0; i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            model_ref [i] = ref_info-> data [i];
        }
    
        for (i = 0; i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            model_deg [i] = deg_info-> data [i];
        }
    
        input_filter( ref_info, deg_info, ftmp );

        printf (" Variable delay compensation...\n");            
        calc_VAD (ref_info);
        calc_VAD (deg_info);
        
        crude_align (ref_info, deg_info, err_info, WHOLE_SIGNAL, ftmp);

        utterance_locate (ref_info, deg_info, err_info, ftmp);
    
        for (i = 0; i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            ref_info-> data [i] = model_ref [i];
        }
    
        for (i = 0; i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            deg_info-> data [i] = model_deg [i];
        }

        safe_free (model_ref);
        safe_free (model_deg); 
    
        if ((*Error_Flag) == 0) {
            if (ref_info-> Nsamples < deg_info-> Nsamples) {
                float *new_ref = (float *) safe_malloc((deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof(float));
                long  i;
                for (i = 0; i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                    new_ref [i] = ref_info-> data [i];
                }
                for (i = ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); 
                     i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                    new_ref [i] = 0.0f;
                }
                safe_free (ref_info-> data);
                ref_info-> data = new_ref;
                new_ref = NULL;
            } else {
                if (ref_info-> Nsamples > deg_info-> Nsamples) {
                    float *new_deg = (float *) safe_malloc((ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof(float));
                    long  i;
                    for (i = 0; i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                        new_deg [i] = deg_info-> data [i];
                    }
                    for (i = deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); 
                         i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                        new_deg [i] = 0.0f;
                    }
                    safe_free (deg_info-> data);
                    deg_info-> data = new_deg;
                    new_deg = NULL;
                }
            }
        }        

        printf (" Acoustic model processing...\n");    
        pesq_psychoacoustic_model (ref_info, deg_info, err_info, ftmp);
    
        safe_free (ref_info-> data);
        safe_free (ref_info-> VAD);
        safe_free (ref_info-> logVAD);
        safe_free (deg_info-> data);
        safe_free (deg_info-> VAD);
        safe_free (deg_info-> logVAD);
        safe_free (ftmp);

        resultsFile = fopen (ITU_RESULTS_FILE, "at");

        if (resultsFile != NULL) {
            long start, end;

            if (0 != fseek (resultsFile, 0, SEEK_SET)) {
                printf ("Could not move to start of results file %s!\n", ITU_RESULTS_FILE);
                exit (1);
            }
            start = ftell (resultsFile);

            if (0 != fseek (resultsFile, 0, SEEK_END)) {
                printf ("Could not move to end of results file %s!\n", ITU_RESULTS_FILE);
                exit (1);
            }
            end = ftell (resultsFile);

            if (start == end) {
                fprintf (resultsFile, "REFERENCE\t DEGRADED\t PESQMOS\t PESQMOS\t SUBJMOS\t COND\t SAMPLE_FREQ\t CRUDE_DELAY\n");
                fflush (resultsFile);
            }

            fprintf (resultsFile, "%s\t ", ref_info-> path_name);
            fprintf (resultsFile, "%s\t ", deg_info-> path_name);
            fprintf (resultsFile, "SQValue=%.3f\t ", err_info->pesq_mos);
            fprintf (resultsFile, "%.3f\t ", err_info->pesq_mos);
            fprintf (resultsFile, "%.3f\t ", err_info->subj_mos);
            fprintf (resultsFile, "%d\t ", err_info->cond_nr);
            fprintf (resultsFile, "%d\t", Fs);
            fprintf (resultsFile, "%.4f\n ", (float) err_info-> Crude_DelayEst / (float) Fs); 
            
            fclose (resultsFile);
        }

        resultsFile = fopen (SIMPLE_RESULTS_FILE, "at");

        if (resultsFile != NULL) {
            long start, end;

            if (0 != fseek (resultsFile, 0, SEEK_SET)) {
                printf ("Could not move to start of results file %s!\n", SIMPLE_RESULTS_FILE);
                exit (1);
            }
            start = ftell (resultsFile);

            if (0 != fseek (resultsFile, 0, SEEK_END)) {
                printf ("Could not move to end of results file %s!\n", SIMPLE_RESULTS_FILE);
                exit (1);
            }
            end = ftell (resultsFile);

            if (start == end) {
                fprintf (resultsFile, "DEGRADED\t PESQMOS\t SUBJMOS\t COND\t SAMPLE_FREQ\t CRUDE_DELAY\n");
                fflush (resultsFile);
            }

            fprintf (resultsFile, "%s\t ", deg_info-> file_name);
            fprintf (resultsFile, "%.3f\t ", err_info->pesq_mos);
            fprintf (resultsFile, "%.3f\t ", err_info->subj_mos);
            fprintf (resultsFile, "%d\t ", err_info->cond_nr);
            fprintf (resultsFile, "%d\t", Fs);
            fprintf (resultsFile, "%.4f\n ", (float) err_info-> Crude_DelayEst / (float) Fs); 
            
            fclose (resultsFile);
        }
    }

    return;
}

/* END OF FILE */
