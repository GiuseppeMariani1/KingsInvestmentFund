"""
STOXX Europe 600 Index Components - Yahoo Finance Tickers
Updated: January 2026

This provides the official STOXX 600 universe for the investment dashboard.
Tickers are formatted for Yahoo Finance (e.g., SHEL.L for London, SAP.DE for Frankfurt)
"""

STOXX_600_TICKERS = [
    # ============ UK (.L) ============
    'BATS.L', 'HSBA.L', 'BARC.L', 'IHG.L', 'BP.L', 'NXT.L', 'LLOY.L', 'MKS.L',
    'BIRG.L', 'NWG.L', 'PRU.L', 'SHEL.L', 'HEIA.L', 'AZN.L', 'BLND.L', 'LGEN.L',
    'IMB.L', 'RKT.L', 'KGF.L', 'REL.L', 'CNA.L', 'III.L', 'WPP.L', 'VCT.L',
    'HL.L', 'TSCO.L', 'ABDN.L', 'DGE.L', 'HIK.L', 'WG.L', 'INVP.L', 'RIO.L',
    'JMAT.L', 'AAL.L', 'CPG.L', 'BNZL.L', 'BRBY.L', 'GSK.L', 'BDEV.L', 'AV.L',
    'BHP.L', 'SGE.L', 'WHS.L', 'PSON.L', 'HMSO.L', 'ABF.L', 'TLW.L', 'LSEG.L',
    'BA.L', 'ANTO.L', 'MNG.L', 'BRV.L', 'RR.L', 'WTB.L', 'PHNX.L', 'TATE.L',
    'PSN.L', 'ADM.L', 'UU.L', 'ULVR.L', 'CRH.L', 'SSE.L', 'STAN.L', 'BLND.L',
    'INF.L', 'WEIR.L', 'ASHM.L', 'SVT.L', 'TPK.L', 'SJP.L', 'SMIN.L', 'CBG.L',
    'IGG.L', 'MRO.L', 'SN.L', 'ITRK.L', 'MNDI.L', 'RTO.L', 'SBRY.L', 'CRDA.L',
    'NG.L', 'IMI.L', 'BT-A.L', 'INC.L', 'SGRO.L', 'SDR.L', 'RNK.L', 'CPP.L',
    'HIK.L', 'HSX.L', 'LAND.L', 'IWG.L', 'SPX.L', 'HLMA.L', 'RMV.L', 'DS.L',
    'TW.L', 'SPX.L', 'RS1.L', 'AHT.L', 'UTG.L', 'DRW.L', 'FLTR.L', 'BEZ.L',
    'FERG.L', 'BLW.L', 'HJO.L', 'DNB.L', 'EDN.L', 'GLEN.L', 'IAG.L', 'EXPN.L',
    'BRBY.L', 'SKG.L', 'DCC.L', 'SMWH.L', 'OCDO.L', 'JD.L', 'JPLF.L', 'LAND.L',
    'SSPG.L', 'BBOX.L', 'SCAT.L', 'SPI.L', 'AUTO.L', 'BME.L', 'CCHG.L', 'GLEN.L',
    'CNVT.L', 'ITG.L', 'PSTL.L', 'FNEX.L', 'SMWH.L', 'ENT.L', 'KGSP.L', 'DLGD.L',
    'QQ.L', 'RNK.L',
    
    # ============ Germany (.DE) ============
    'SAP.DE', 'SIE.DE', 'ALV.DE', 'MUV2.DE', 'DTE.DE', 'BAS.DE', 'BAYN.DE',
    'BMW.DE', 'MBG.DE', 'VOW3.DE', 'ADS.DE', 'AIR.DE', 'DBK.DE', 'RWE.DE',
    'IFX.DE', 'HEN3.DE', 'DHL.DE', 'EOAN.DE', 'FRE.DE', 'CON.DE', 'DB1.DE',
    'MTX.DE', 'SHL.DE', 'BEI.DE', 'HEI.DE', 'MRK.DE', 'PUM.DE', 'QIA.DE',
    'RHM.DE', 'SY1.DE', 'VNA.DE', 'ZAL.DE', 'LXS.DE', 'FME.DE', 'FRE.DE',
    'FNT.DE', 'G1A.DE', 'HNR1.DE', 'KWS.DE', 'KSB.DE', 'SZG.DE', 'EVT.DE',
    'NEM.DE', 'SAR.DE', 'DWS.DE', 'LEG.DE', 'TLX.DE', 'EVK.DE', 'SAG.DE',
    'CBK.DE', 'BOSS.DE', 'BNR.DE', 'TAG.DE', 'SIX2.DE', 'SRT3.DE', 'BC8.DE',
    'KIO.DE', 'KBX.DE', 'HLE.DE', 'DEL.DE', 'SGL.DE', 'PSM.DE', 'S92.DE',
    'UN01.DE', 'MOR.DE', 'CTS.DE', 'GRN.DE', 'G24.DE', 'NDA.DE',
    
    # ============ France (.PA) ============
    'MC.PA', 'OR.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'SU.PA', 'AI.PA',
    'CS.PA', 'SAF.PA', 'BN.PA', 'RI.PA', 'KER.PA', 'CAP.PA', 'DG.PA', 'ENGI.PA',
    'RMS.PA', 'EL.PA', 'GLE.PA', 'ACA.PA', 'ORA.PA', 'STLAP.PA', 'ML.PA',
    'SGO.PA', 'VIE.PA', 'DSY.PA', 'PUB.PA', 'LR.PA', 'EN.PA', 'CA.PA', 'VIV.PA',
    'STM.PA', 'DG.PA', 'RNO.PA', 'VK.PA', 'ADP.PA', 'EDF.PA', 'ATO.PA', 'ALO.PA',
    'UBI.PA', 'SW.PA', 'NK.PA', 'ERA.PA', 'SOP.PA', 'RUI.PA', 'EDR.PA', 'IPN.PA',
    'JCQ.PA', 'BIM.PA', 'TFI.PA', 'BVI.PA', 'AKE.PA', 'SK.PA', 'ALT.PA', 'LI.PA',
    'KLE.PA', 'GFC.PA', 'COV.PA', 'ETE.PA', 'WLN.PA', 'SESG.PA', 'ERF.PA',
    'BOL.PA', 'GET.PA', 'ABI.PA', 'ELS.PA', 'SRP.PA', 'SPIE.PA', 'AMA.PA',
    
    # ============ Netherlands (.AS) ============
    'ASML.AS', 'INGA.AS', 'PHIA.AS', 'AD.AS', 'UNA.AS', 'HEIA.AS', 'WKL.AS',
    'PRX.AS', 'ADYEN.AS', 'ASM.AS', 'AKZA.AS', 'RAND.AS', 'NN.AS', 'SBM.AS',
    'VPK.AS', 'ABN.AS', 'MT.AS', 'KPN.AS', 'AALB.AS', 'PRY.AS', 'SIGN.AS',
    'ASRNL.AS', 'CRBN.AS', 'JDEP.AS', 'TKWY.AS', 'IMCD.AS', 'ENX.AS', 'AGN.AS',
    
    # ============ Switzerland (.SW) ============
    'NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW', 'ZURN.SW', 'ABBN.SW', 'CFR.SW',
    'SREN.SW', 'HOLN.SW', 'SIKA.SW', 'GIVN.SW', 'LONN.SW', 'GEBN.SW', 'SCMN.SW',
    'ADEN.SW', 'BALN.SW', 'LOGN.SW', 'CLN.SW', 'SLHN.SW', 'BARN.SW', 'SGSN.SW',
    'SOON.SW', 'SWTQ.SW', 'PGHN.SW', 'SCHP.SW', 'KNIN.SW', 'LINP.SW', 'HELN.SW',
    'PSPS.SW', 'VACN.SW', 'TEMN.SW', 'SRAIL.SW', 'SIGN.SW', 'ALLN.SW', 'FHZN.SW',
    'TECN.SW', 'EMSN.SW', 'OERL.SW', 'DUFN.SW', 'BELM.SW', 'STMN.SW', 'GALM.SW',
    'CMBN.SW', 'VATS.SW', 'GF.SW', 'FHZN.SW',
    
    # ============ Spain (.MC) ============
    'SAN.MC', 'IBE.MC', 'ITX.MC', 'TEF.MC', 'BBVA.MC', 'AMS.MC', 'FER.MC',
    'CABK.MC', 'REP.MC', 'ELE.MC', 'AENA.MC', 'GRF.MC', 'MAP.MC', 'MEL.MC',
    'ACS.MC', 'BKT.MC', 'SAB.MC', 'ENG.MC', 'RED.MC', 'COL.MC', 'CLNX.MC',
    
    # ============ Italy (.MI) ============
    'ENEL.MI', 'ISP.MI', 'ENI.MI', 'UCG.MI', 'RACE.MI', 'G.MI', 'PRY.MI',
    'MONC.MI', 'TEN.MI', 'SPM.MI', 'TRN.MI', 'PIRC.MI', 'A2A.MI', 'SRS.MI',
    'HER.MI', 'IP.MI', 'LDO.MI', 'SAI.MI', 'BAMI.MI', 'BMED.MI', 'AMP.MI',
    'REC.MI', 'NEXI.MI', 'PST.MI', 'IG.MI', 'IWB.MI', 'FBK.MI', 'DIA.MI',
    
    # ============ Nordic - Sweden (.ST) ============
    'ATCO-A.ST', 'VOLV-B.ST', 'ERIC-B.ST', 'HM-B.ST', 'INVE-B.ST', 'SAND.ST',
    'NDA-SE.ST', 'SHB-A.ST', 'HEXA-B.ST', 'ESSITY-B.ST', 'ELUX-B.ST', 'ALFA.ST',
    'BOL.ST', 'ATCO-B.ST', 'ASSA-B.ST', 'SEB-A.ST', 'SKA-B.ST', 'SWED-A.ST',
    'SKF-B.ST', 'TEL2-B.ST', 'TELIA.ST', 'SCV-A.ST', 'SECU-B.ST', 'LUND-B.ST',
    'SAAB-B.ST', 'TREL-B.ST', 'KINV-B.ST', 'INDT.ST', 'AAK.ST', 'AFRY.ST',
    'BALD-B.ST', 'BEIJ-B.ST', 'HPOL-B.ST', 'INDU-A.ST', 'NIBE-B.ST', 'SOBI.ST',
    'CAST.ST', 'FAB.ST', 'GETI-B.ST', 'HOLL.ST', 'HUSQ-B.ST', 'EVO.ST',
    'DOM.ST', 'LEO.ST', 'SAGA-B.ST', 'EPI-A.ST', 'EQT.ST',
    
    # ============ Nordic - Denmark (.CO) ============
    'NOVO-B.CO', 'MAERSK-B.CO', 'ORSTED.CO', 'CARL-B.CO', 'VWS.CO', 'DSV.CO',
    'DANSKE.CO', 'NZYM-B.CO', 'GMAB.CO', 'GN.CO', 'TRYG.CO', 'COLO-B.CO',
    'TOP.CO', 'DEMANT.CO', 'AMBU-B.CO', 'PNDORA.CO', 'RBREW.CO', 'ISS.CO',
    
    # ============ Nordic - Finland (.HE) ============
    'NOKIA.HE', 'SAMPO.HE', 'KNEBV.HE', 'UPM.HE', 'FORTUM.HE', 'NESTE.HE',
    'ELISA.HE', 'KESKO.HE', 'STERV.HE', 'NRE1V.HE', 'WRT1V.HE', 'ORNAV.HE',
    'HUH1V.HE', 'VALMT.HE', 'KOJAMO.HE',
    
    # ============ Nordic - Norway (.OL) ============
    'EQNR.OL', 'DNB.OL', 'TEL.OL', 'NHY.OL', 'YAR.OL', 'MOWI.OL', 'ORK.OL',
    'TOM.OL', 'STB.OL', 'SUBC.OL', 'SALM.OL', 'SCHA.OL', 'GJF.OL', 'SCATC.OL',
    'AKRBP.OL', 'NEL.OL',
    
    # ============ Belgium (.BR) ============
    'ABI.BR', 'KBC.BR', 'UCB.BR', 'AGS.BR', 'SOLB.BR', 'ACKB.BR', 'PROX.BR',
    'COFB.BR', 'GBL.BR', 'UMI.BR', 'AED.BR', 'ELIA.BR', 'GLPG.BR', 'SOF.BR',
    'WDP.BR',
    
    # ============ Ireland (.IR / .L) ============
    'AIB.L', 'KYGA.L', 'FLT.L', 'GLEN.L', 'SKG.L', 'KRX.L',
    
    # ============ Austria (.VI) ============
    'ANA.VI', 'EBS.VI', 'IIA.VI', 'OMV.VI', 'RBI.VI', 'VOE.VI', 'VER.VI', 'WIE.VI',
    
    # ============ Portugal (.LS) ============
    'GALP.LS', 'JMT.LS', 'EDP.LS',
    
    # ============ Poland (.WA) ============
    'SPL.WA', 'PKO.WA', 'PEO.WA', 'KGH.WA', 'PKN.WA', 'PZU.WA', 'CDR.WA', 'DNP.WA',
    'ALG.WA',
]

# Remove duplicates and sort
STOXX_600_TICKERS = sorted(list(set(STOXX_600_TICKERS)))

def get_stoxx600_tickers():
    """Return the STOXX 600 ticker list"""
    return STOXX_600_TICKERS.copy()

def get_tickers_by_country():
    """Return tickers grouped by country/exchange"""
    groups = {
        'UK': [t for t in STOXX_600_TICKERS if t.endswith('.L')],
        'Germany': [t for t in STOXX_600_TICKERS if t.endswith('.DE')],
        'France': [t for t in STOXX_600_TICKERS if t.endswith('.PA')],
        'Netherlands': [t for t in STOXX_600_TICKERS if t.endswith('.AS')],
        'Switzerland': [t for t in STOXX_600_TICKERS if t.endswith('.SW')],
        'Spain': [t for t in STOXX_600_TICKERS if t.endswith('.MC')],
        'Italy': [t for t in STOXX_600_TICKERS if t.endswith('.MI')],
        'Sweden': [t for t in STOXX_600_TICKERS if t.endswith('.ST')],
        'Denmark': [t for t in STOXX_600_TICKERS if t.endswith('.CO')],
        'Finland': [t for t in STOXX_600_TICKERS if t.endswith('.HE')],
        'Norway': [t for t in STOXX_600_TICKERS if t.endswith('.OL')],
        'Belgium': [t for t in STOXX_600_TICKERS if t.endswith('.BR')],
        'Austria': [t for t in STOXX_600_TICKERS if t.endswith('.VI')],
        'Portugal': [t for t in STOXX_600_TICKERS if t.endswith('.LS')],
        'Poland': [t for t in STOXX_600_TICKERS if t.endswith('.WA')],
    }
    return groups

if __name__ == "__main__":
    print(f"Total STOXX 600 tickers: {len(STOXX_600_TICKERS)}")
    groups = get_tickers_by_country()
    for country, tickers in groups.items():
        print(f"  {country}: {len(tickers)} stocks")
