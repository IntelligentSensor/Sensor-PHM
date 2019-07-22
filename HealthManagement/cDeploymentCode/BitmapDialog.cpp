//
// BitmapDialog.cpp
//

#include "stdafx.h"
#include "MainCtrlPanel.h"
#include "BitmapDialog.h"
#include "resource.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CBitmapDialog dialog

CBitmapDialog::CBitmapDialog()
{
	CommonConstruct();
}


CBitmapDialog::CBitmapDialog(UINT uResource, CWnd* pParent /*=NULL*/)
	: CDialog(uResource, pParent)
{
	CommonConstruct();
}


CBitmapDialog::CBitmapDialog(LPCTSTR pszResource, CWnd* pParent /*=NULL*/)
	: CDialog(pszResource, pParent)
{
	CommonConstruct();
}

CBitmapDialog::~CBitmapDialog()
{
	if ( m_TitleFont.GetSafeHandle() != NULL )
	{
		m_TitleFont.DeleteObject();
	}
	if ( m_BodyFont.GetSafeHandle() != NULL )
	{
		m_BodyFont.DeleteObject();
	}
	if ( m_HollowBrush.GetSafeHandle() != NULL )
	{
		m_HollowBrush.DeleteObject();
	}
	if ( m_BlaskBrush.GetSafeHandle() != NULL )
	{
		m_BlaskBrush.DeleteObject();
	}
	if ( m_DlineBrush.GetSafeHandle() != NULL )
	{
		m_DlineBrush.DeleteObject();
	}
}

void CBitmapDialog::CommonConstruct()
{
	m_FontSize = 0;
	bInilialize = FALSE;
	bFirstErase = TRUE;
	nTimeTimerCount = -1;
    bDisplayTime = FALSE;
	bHasDline = FALSE;
	bHasDisplayWindow = FALSE;
	bHasTitle = FALSE;
	bHasBody = FALSE;
    m_DCBackground = NULL;
	m_nType = BITMAP_TILE;
	m_HollowBrush.CreateStockObject(HOLLOW_BRUSH);
	m_BlaskBrush.CreateStockObject(BLACK_BRUSH);
	m_DlineBrush.CreateSolidBrush(DIVIDING_LINES_COLOR);
	mBodyColor = BODY_TEXT_COLOR;
	mBodyNumColor = BODY_NUMBER_COLOR;

	//{{AFX_DATA_INIT(CBitmapDialog)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
}


BOOL CBitmapDialog :: SetBitmap(HDC hdc, int x, int y, int w, int h, int nType) {
	m_nType = nType;
	nBitmapXpos =x;
	nBitmapYpos =y;
    nBitmapWidth =w;
    nBitmapHight =abs(h);
	m_DCBackground = hdc;
	return TRUE;
}


void CBitmapDialog::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CBitmapDialog)
		// NOTE: the ClassWizard will add DDX and DDV calls here
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CBitmapDialog, CDialog)
	//{{AFX_MSG_MAP(CBitmapDialog)
	ON_WM_ERASEBKGND()
	ON_WM_CTLCOLOR()
	ON_WM_QUERYNEWPALETTE()
	ON_WM_PALETTECHANGED()
	ON_WM_TIMER()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CBitmapDialog message handlers
BOOL CBitmapDialog::IsDigit(TCHAR ch)
{
    static BOOL bNumber = FALSE;

    if((ch>='0')&&(ch<='9'))
    {
        bNumber=TRUE;
    }
    else if((bNumber) && (ch=='.'))
    {
        bNumber=TRUE;
    }
    else
    {
        bNumber=FALSE;
    }

    return bNumber;
}

BOOL CBitmapDialog::OnEraseBkgnd(CDC* pDC) 
{
	HDC hBKDC;
	if(bFirstErase)
	{
		bFirstErase = FALSE;
	}
	hBKDC = pDC->m_hDC;

	if(m_DCBackground)
	{
		ASSERT(m_nType == BITMAP_TILE || m_nType == BITMAP_STRETCH || m_nType == BITMAP_CENTER);
		
		CRect rc;
		GetClientRect(rc);
		int x = 0, y = 0;

		switch(m_nType) {
			case BITMAP_CENTER:
				// center the bitmap
				//CDialog::OnEraseBkgnd(pDC);
				x = (rc.Width() - nBitmapWidth) / 2;
				y = (rc.Height() - nBitmapHight) / 2;
				::BitBlt(hBKDC, 0, 0, nBitmapWidth, nBitmapHight, m_DCBackground, nBitmapXpos, nBitmapYpos, SRCCOPY);
				break;

			case BITMAP_STRETCH:
			    ::StretchBlt( hBKDC, 0,0,rc.Width(), rc.Height(),
								m_DCBackground, nBitmapXpos, nBitmapYpos, nBitmapWidth, nBitmapHight,SRCCOPY);
				break;

			case BITMAP_TILE:
			default:
				while(y < rc.Height()) 
				{
					while(x < rc.Width()) 
					{
					    ::BitBlt(hBKDC, x, y, nBitmapWidth, nBitmapHight, m_DCBackground, nBitmapXpos, nBitmapYpos,SRCCOPY);
						x += nBitmapWidth;
					}
					x = 0;
					y += nBitmapHight;
				}
				break;
		}

		if((bDisplayTime) && (nTimeTimerCount>0))
		{
			DisplayTime();
		}
		if(bHasDline)
		{
			CRect rectl(DIVIDING_LINES_X,
				DIVIDING_LINES_Y,
				DIVIDING_LINES_X+DIVIDING_LINES_W,
				DIVIDING_LINES_Y+DIVIDING_LINES_H);

			pDC->FillRect(&rectl, &m_DlineBrush);
		}

		if(bHasTitle)
		{
			HFONT hOldFont = NULL;
			TCHAR szTitle[256];
			RECT rect;
			if ( m_TitleFont.GetSafeHandle() != NULL )
			{
				hOldFont = ( HFONT ) pDC->SelectObject( m_TitleFont.GetSafeHandle() );
			}
			rect.left=0;
			rect.top=4;
			rect.bottom=GetSystemMetrics(SM_CYSCREEN);
			rect.right=GetSystemMetrics(SM_CXSCREEN);
			GetWindowText(szTitle, 256);
			pDC->SetBkMode( TRANSPARENT );
			pDC->SetTextColor( TITLE_TEXT_COLOR );
			pDC->DrawText(szTitle, _tcslen(szTitle), &rect, DT_SINGLELINE | DT_CENTER );
			if ( hOldFont != NULL )
			{
				pDC->SelectObject( hOldFont );
			}
		}

	    if(bHasBody)
		{
            DWORD dwTextStyle;
            CWnd* pItemWnd;
			HFONT hOldFont = NULL;
			TCHAR szBody[256];
			TCHAR szChar[2];
			RECT rect;
			RECT rectrl;
			int nTextID, nH1, nH2, n;

			if ( m_BodyFont.GetSafeHandle() != NULL )
			{
				hOldFont = ( HFONT ) pDC->SelectObject( m_BodyFont.GetSafeHandle() );
			}
			pDC->SetBkMode( TRANSPARENT );
			pDC->SetTextColor( mBodyColor );
			for(nTextID=IDC_BODY_TEXT0; nTextID<(IDC_BODY_TEXT0+10); nTextID++)
			{
                pItemWnd=GetDlgItem(nTextID);
				if(pItemWnd==NULL) continue;
				if(pItemWnd->GetWindowText(szBody, 256 )<=0) continue;
	    		pItemWnd->GetWindowRect(&rect);
				pItemWnd->GetWindowText(szBody, 256);
				ScreenToClient(&rect);
                memcpy(&rectrl, &rect, sizeof(RECT));
				nH1=rect.bottom -rect.top;
            
                dwTextStyle=GetWindowLong(pItemWnd->m_hWnd, GWL_USERDATA);
				if(dwTextStyle==TEXT_STYLE_NORMAL)
				{
					pDC->DrawText(szBody, &rect, DT_WORDBREAK | DT_EDITCONTROL | DT_TOP | DT_LEFT );
				}
				else if(dwTextStyle==TEXT_STYLE_VCENTER)
				{
					pDC->DrawText(szBody, &rect, DT_CALCRECT | DT_WORDBREAK | DT_EDITCONTROL | DT_TOP | DT_LEFT );
					nH2=rect.bottom -rect.top;
					rect.top = rect.top+(nH1-nH2)/2;
					rect.bottom = rect.bottom+(nH1-nH2)/2;
					pDC->DrawText(szBody, &rect, DT_WORDBREAK | DT_EDITCONTROL | DT_TOP | DT_LEFT );
				}
				else if(dwTextStyle==TEXT_STYLE_NUM_HIGHLIGHT)
				{
                    pDC->DrawText(TEXT("A"), &rectrl, DT_CALCRECT | DT_TOP | DT_LEFT );
                    nH1=rectrl.bottom -rectrl.top;
                    nH2=rectrl.left;
                    for(n=0;n<(int)_tcslen(szBody);n++)
                    {
                        szChar[1]=0;
                        szChar[0]=szBody[n];
                        if(szChar[0]=='\r')
                        {
                            continue;
                        }
                        else if(szChar[0]=='\n')
                        {
                            rectrl.top += nH1;
                            rectrl.bottom += nH1;
                            rectrl.left = nH2;
                        }
                        else if(IsDigit(szChar[0]))
                        {
                            pDC->DrawText(szChar, &rect, DT_CALCRECT | DT_TOP | DT_LEFT );
                            rectrl.right = rectrl.left+(rect.right-rect.left);
                            pDC->SetTextColor( mBodyNumColor );
                            pDC->DrawText(szChar, &rectrl, DT_TOP | DT_LEFT );
                            rectrl.left=rectrl.right;
                        }
                        else
                        {
                            pDC->DrawText(szChar, &rect, DT_CALCRECT | DT_TOP | DT_LEFT );
                            rectrl.right = rectrl.left+(rect.right-rect.left);
                            pDC->SetTextColor( mBodyColor );
                            pDC->DrawText(szChar, &rectrl, DT_TOP | DT_LEFT );
                            rectrl.left=rectrl.right;
                        }
                    }
				}
			}
			if ( hOldFont != NULL )
			{
				pDC->SelectObject( hOldFont );
			}
		}

		if(bHasDisplayWindow)
		{
		}
   		return TRUE;
	}	
	//pDC->DeleteTempMap();
	return CDialog::OnEraseBkgnd(pDC);
}

void CBitmapDialog::SetTimeFont( CString srtFntName_i, int nSize_i )
{
    LOGFONT lfCtrl = {0};
	if ( m_TimeFont.GetSafeHandle() != NULL )
	{
		if(m_FontSize == nSize_i)
			return;
		m_TimeFont.DeleteObject();
	}
	m_FontSize = nSize_i;
    lfCtrl.lfOrientation = 0 ;
    lfCtrl.lfEscapement = 0 ;

    lfCtrl.lfHeight = nSize_i;

    lfCtrl.lfItalic = FALSE;
    lfCtrl.lfUnderline = FALSE;
    lfCtrl.lfStrikeOut = FALSE;

    lfCtrl.lfCharSet = DEFAULT_CHARSET;
    lfCtrl.lfQuality = CLEARTYPE_COMPAT_QUALITY;
    lfCtrl.lfOutPrecision = OUT_DEFAULT_PRECIS;
    lfCtrl.lfPitchAndFamily = FIXED_PITCH;
    _tcscpy( lfCtrl.lfFaceName, srtFntName_i.GetBuffer() );
    lfCtrl.lfWeight = FW_MEDIUM;
    m_TimeFont.CreateFontIndirect( &lfCtrl );
    srtFntName_i.ReleaseBuffer();
}

void CBitmapDialog::SetTitle( CString srtFntName_i)
{
	SetWindowText(srtFntName_i);
	SetTitleFont(TEXT("宋体"), 24);
}

void CBitmapDialog::SetTitleFont( CString srtFntName_i, int nSize_i, COLORREF mColor )
{
    LOGFONT lfCtrl = {0};
	if ( m_TitleFont.GetSafeHandle() != NULL )
	{
		return;
	}
    lfCtrl.lfOrientation = 0 ;
    lfCtrl.lfEscapement = 0 ;

    lfCtrl.lfHeight = nSize_i;

    lfCtrl.lfItalic = FALSE;
    lfCtrl.lfUnderline = FALSE;
    lfCtrl.lfStrikeOut = FALSE;

    lfCtrl.lfCharSet = DEFAULT_CHARSET;
    lfCtrl.lfQuality = CLEARTYPE_COMPAT_QUALITY;
    lfCtrl.lfOutPrecision = OUT_DEFAULT_PRECIS;
    lfCtrl.lfPitchAndFamily = DEFAULT_PITCH;
    _tcscpy( lfCtrl.lfFaceName, srtFntName_i.GetBuffer() );
    lfCtrl.lfWeight = FW_MEDIUM;
    m_TitleFont.CreateFontIndirect( &lfCtrl );
    srtFntName_i.ReleaseBuffer();
}

int CBitmapDialog::GetBodyStringDrawWidth(CString mstr)
{
    RECT rectrl={0};
    HDC hdc = ::GetDC(0);
	HFONT hOldFont = NULL;

	if ( m_BodyFont.GetSafeHandle() != NULL )
	{
        hOldFont = ( HFONT ) ::SelectObject(hdc,  m_BodyFont.GetSafeHandle() );
	}

    ::DrawText(hdc, mstr, mstr.GetLength(), &rectrl, DT_CALCRECT | DT_TOP | DT_LEFT );

	if ( hOldFont != NULL )
	{
        ::SelectObject( hdc, hOldFont );
	}

    ::ReleaseDC(0, hdc);

    return rectrl.right-rectrl.left;
}

void CBitmapDialog::SetBodyFont( CString srtFntName_i, int nSize_i, COLORREF mColor, COLORREF mNumColor)
{
    LOGFONT lfCtrl = {0};
	if ( m_BodyFont.GetSafeHandle() != NULL )
	{
		return;
	}
	mBodyColor = mColor;
	mBodyNumColor = mNumColor;
    lfCtrl.lfOrientation = 0 ;
    lfCtrl.lfEscapement = 0 ;

    lfCtrl.lfHeight = nSize_i;

    lfCtrl.lfItalic = FALSE;
    lfCtrl.lfUnderline = FALSE;
    lfCtrl.lfStrikeOut = FALSE;

    lfCtrl.lfCharSet = DEFAULT_CHARSET;
    lfCtrl.lfQuality = CLEARTYPE_COMPAT_QUALITY;
    lfCtrl.lfOutPrecision = OUT_DEFAULT_PRECIS;
    lfCtrl.lfPitchAndFamily = DEFAULT_PITCH;
    _tcscpy( lfCtrl.lfFaceName, srtFntName_i.GetBuffer() );
    lfCtrl.lfWeight = FW_MEDIUM;
    m_BodyFont.CreateFontIndirect( &lfCtrl );
    srtFntName_i.ReleaseBuffer();
}

void CBitmapDialog::PostNcDestroy()
{
	CDialog::PostNcDestroy();
	DeleteTempMap();
	AfxLockTempMaps();
	AfxUnlockTempMaps(TRUE);
}

HBRUSH CBitmapDialog::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor) 
{
	if(m_DCBackground) {
		switch(nCtlColor) {
			case CTLCOLOR_STATIC:
				pDC->SetBkMode(2);
				pDC->SetBkColor(RGB(255,255,255));
				return HBRUSH(m_BlaskBrush);

			case CTLCOLOR_BTN:
				// let static controls shine through
				pDC->SetBkMode(TRANSPARENT);
				return HBRUSH(m_HollowBrush);

			default:
				break;
		}
	}
	return CDialog::OnCtlColor(pDC, pWnd, nCtlColor);
}

void CBitmapDialog::InitDisplayTime()
{
	CDC* pDC = GetDC();

	if(!mTimeDisplayMemDC.GetSafeHdc())
	{
		mTimeDisplayMemDC.CreateCompatibleDC(pDC);
	}
	if(!mTimeDisplayBKDC.GetSafeHdc())
	{
		mTimeDisplayBKDC.CreateCompatibleDC(pDC);
	}

	if(!mTimeDisplayMemBitmap.GetSafeHandle())
	{
		mTimeDisplayMemBitmap.CreateCompatibleBitmap(pDC, 
			mTimeDisplayRect.Width(), 
			mTimeDisplayRect.Height());
	}
	mTimeDisplayMemDC.SelectObject(mTimeDisplayMemBitmap);

	if(!mTimeDisplayBKBitmap.GetSafeHandle())
	{
		mTimeDisplayBKBitmap.CreateCompatibleBitmap(pDC, 
			mTimeDisplayRect.Width(), 
			mTimeDisplayRect.Height());
	}
	mTimeDisplayBKDC.SelectObject(mTimeDisplayBKBitmap);

	mTimeDisplayBKDC.BitBlt(0,0,mTimeDisplayRect.Width(), mTimeDisplayRect.Height(),
		pDC, mTimeDisplayRect.left , mTimeDisplayRect.top, SRCCOPY);
	ReleaseDC(pDC);
}

const TCHAR* szWeekDayChs[] = 
{
	TEXT("星期日"),
	TEXT("星期一"),
	TEXT("星期二"),
	TEXT("星期三"),
	TEXT("星期四"),
	TEXT("星期五"),
	TEXT("星期六"),
};

void CBitmapDialog::DisplayTime()
{
	RECT rect;
	SYSTEMTIME mLocalTime;
	HFONT hOldFont = NULL;
	TCHAR szTime[256];

	CDC* pDC = GetDC();
	
	::GetLocalTime(&mLocalTime);
#if 0
	_stprintf(szTime, TEXT("%04d-%02d-%02d %s %02d:%02d:%02d\0"), 
		mLocalTime.wYear,
		mLocalTime.wMonth,
		mLocalTime.wDay,
		szWeekDayChs[mLocalTime.wDayOfWeek],
		mLocalTime.wHour,
		mLocalTime.wMinute,
		mLocalTime.wSecond
		);
#else
	_stprintf(szTime, TEXT("%04d/%02d/%02d %02d:%02d:%02d\0"), 
		mLocalTime.wYear,
		mLocalTime.wMonth,
		mLocalTime.wDay,
		mLocalTime.wHour,
		mLocalTime.wMinute,
		mLocalTime.wSecond
		);
#endif
	mTimeDisplayMemDC.SetBkMode( TRANSPARENT );
    mTimeDisplayMemDC.SetTextColor( RGB(255,255,255) );
    if ( m_TimeFont.GetSafeHandle() != NULL )
    {
        hOldFont = ( HFONT ) mTimeDisplayMemDC.SelectObject( m_TimeFont.GetSafeHandle() );
    }
	mTimeDisplayMemDC.BitBlt(0,0,mTimeDisplayRect.Width(), mTimeDisplayRect.Height(),
		&mTimeDisplayBKDC, 0, 0, SRCCOPY);
	rect.left=0;
	rect.top=0;
	rect.bottom=mTimeDisplayRect.Height();
	rect.right=mTimeDisplayRect.Width();
	mTimeDisplayMemDC.DrawText(szTime, _tcslen(szTime), &rect, DT_SINGLELINE | DT_CENTER );
    if ( hOldFont != NULL )
    {
        mTimeDisplayMemDC.SelectObject( hOldFont );
    }
	pDC->BitBlt(mTimeDisplayRect.left , mTimeDisplayRect.top,
		mTimeDisplayRect.Width(), mTimeDisplayRect.Height(),
		&mTimeDisplayMemDC,0,0, SRCCOPY);
	ReleaseDC(pDC);

}

BOOL CBitmapDialog::OnInitDialog()
{
	CDialog::OnInitDialog();

    if((bDisplayTime) && (GetDlgItem(IDC_TIME_DISPLAY)!=NULL))
    {
    	SetTimeFont(TEXT("Arial"),20);

    	GetDlgItem(IDC_TIME_DISPLAY)->GetWindowRect(mTimeDisplayRect);
        
    	SetTimer(0x123, 500, NULL);
    }
	return TRUE;
}

void CBitmapDialog::OnTimer(UINT_PTR nIDEvent)
{
	if((nIDEvent == 0x123) && bDisplayTime)
	{
		if(nTimeTimerCount == -1)
		{
			InitDisplayTime();
		}
		else
		{
			if(IsWindowVisible())
			{
				DisplayTime();
			}
		}
		nTimeTimerCount ++;
	}
	CDialog::OnTimer(nIDEvent);
}

